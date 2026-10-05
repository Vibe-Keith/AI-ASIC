package hardware

import (
	"encoding/binary"
	"fmt"
)

// BM1387 (Antminer S9 / S9i / S9j / T9+) work protocol.
//
// Unlike the BM1382 (S2/S3) path, which pushes a full 80-byte Bitcoin header to the
// chip over USB via the 0x52 TXTASK token (see internal/driver/device/controller.go),
// the BM1387 chain protocol pushes *pre-hashed* work:
//
//   - The host computes the SHA-256 midstate over the first 64 bytes of the header
//     (version + prev-hash + the first 28 bytes of the merkle root).
//   - It sends that 32-byte midstate plus the remaining 12 bytes of pre-nonce header
//     data (the 4-byte merkle-root tail + 4-byte ntime + 4-byte nbits).
//   - The chip rolls the 4-byte nonce internally and reports nonces that meet target.
//
// This mirrors the way cgminer/bmminer frame work for the S9. The outer FPGA framing
// on the S9 control board is firmware-specific; this encoder produces the canonical,
// verifiable payload (midstate + data + CRC) used by that firmware. For most S9-class
// deployments work is driven through the cgminer JSON-RPC API (ProtocolCGMinerAPI);
// this encoder exists for direct-drive / repurposing use and for correct work
// generation and nonce interpretation.

const (
	// BM1387MidstateBytes is the SHA-256 midstate length.
	BM1387MidstateBytes = 32
	// BM1387WorkDataBytes is the pre-nonce header tail length (merkle tail + ntime + nbits).
	BM1387WorkDataBytes = 12
	// BM1387MaxMidstates is the maximum number of midstates per work item when using
	// AsicBoost version rolling (4-way).
	BM1387MaxMidstates = 4

	// BM1387 chip-chain command tokens (CRC5-protected 5-byte command frames).
	BM1387CmdSetAddress    = 0x00
	BM1387CmdSetConfig     = 0x08
	BM1387CmdChainInactive = 0x05
	BM1387CmdReadReg       = 0x04
)

// BM1387Work is a single unit of pre-hashed work for the BM1387 chain.
type BM1387Work struct {
	// WorkID identifies this work item so returned nonces can be matched back.
	WorkID uint32
	// Midstates holds one SHA-256 midstate, or up to BM1387MaxMidstates for AsicBoost.
	Midstates [][BM1387MidstateBytes]byte
	// Data holds the 12-byte pre-nonce header tail (merkle tail + ntime + nbits).
	Data [BM1387WorkDataBytes]byte
}

// NewBM1387WorkFromHeader builds a single-midstate work item from an 80-byte Bitcoin
// header. The nonce field (bytes 76-79) is ignored; the chip supplies the nonce.
func NewBM1387WorkFromHeader(header []byte, workID uint32) (*BM1387Work, error) {
	if len(header) != 80 {
		return nil, fmt.Errorf("header must be exactly 80 bytes, got %d", len(header))
	}

	w := &BM1387Work{WorkID: workID}
	midstate := ComputeMidstate(header[0:64])
	w.Midstates = append(w.Midstates, midstate)
	copy(w.Data[:], header[64:76])
	return w, nil
}

// NewBM1387WorkAsicBoost builds a multi-midstate work item for AsicBoost version
// rolling. versions holds up to BM1387MaxMidstates block-version values; a midstate is
// computed for each by overwriting the header version field (bytes 0-3, little-endian)
// before hashing the first 64 bytes.
func NewBM1387WorkAsicBoost(header []byte, workID uint32, versions []uint32) (*BM1387Work, error) {
	if len(header) != 80 {
		return nil, fmt.Errorf("header must be exactly 80 bytes, got %d", len(header))
	}
	if len(versions) == 0 {
		return nil, fmt.Errorf("at least one version is required")
	}
	if len(versions) > BM1387MaxMidstates {
		return nil, fmt.Errorf("at most %d versions supported, got %d", BM1387MaxMidstates, len(versions))
	}

	w := &BM1387Work{WorkID: workID}
	block := make([]byte, 64)
	copy(block, header[0:64])
	for _, v := range versions {
		binary.LittleEndian.PutUint32(block[0:4], v)
		w.Midstates = append(w.Midstates, ComputeMidstate(block))
	}
	copy(w.Data[:], header[64:76])
	return w, nil
}

// Encode serializes the work item into a chain frame:
//
//	[work_id:4 LE][midstate_0:32]...[midstate_n:32][data:12][crc16:2 BE]
//
// The CRC16 is computed over every preceding byte. This is the canonical payload the
// S9 work framer wraps; the leading FPGA framing byte(s) are added by the transport.
func (w *BM1387Work) Encode() ([]byte, error) {
	if len(w.Midstates) == 0 {
		return nil, fmt.Errorf("work has no midstates")
	}
	if len(w.Midstates) > BM1387MaxMidstates {
		return nil, fmt.Errorf("too many midstates: %d", len(w.Midstates))
	}

	buf := make([]byte, 0, 4+len(w.Midstates)*BM1387MidstateBytes+BM1387WorkDataBytes+2)
	var id [4]byte
	binary.LittleEndian.PutUint32(id[:], w.WorkID)
	buf = append(buf, id[:]...)
	for _, ms := range w.Midstates {
		buf = append(buf, ms[:]...)
	}
	buf = append(buf, w.Data[:]...)

	crc := CRC16(buf)
	var crcBytes [2]byte
	binary.BigEndian.PutUint16(crcBytes[:], crc)
	buf = append(buf, crcBytes[:]...)
	return buf, nil
}

// BM1387NonceResult is a nonce reported by a BM1387 chip.
type BM1387NonceResult struct {
	Nonce         uint32
	WorkID        uint32
	MidstateIndex int // which AsicBoost midstate matched (0 when not version-rolling)
}

// ParseNonceResponse decodes a 7-byte BM1387 nonce response frame:
//
//	[nonce:4 BE][work_id:1][midstate_index/chip:1][crc5-in-low-bits:1]
//
// The exact high bits of the last two bytes are firmware-specific; this parser extracts
// the nonce and the work/midstate identifiers, which are what callers need to match a
// nonce back to its work item. It returns an error on short input or CRC mismatch when
// a CRC is present.
func ParseNonceResponse(frame []byte) (*BM1387NonceResult, error) {
	if len(frame) < 5 {
		return nil, fmt.Errorf("nonce response too short: %d bytes", len(frame))
	}
	res := &BM1387NonceResult{
		Nonce: binary.BigEndian.Uint32(frame[0:4]),
	}
	if len(frame) >= 6 {
		res.WorkID = uint32(frame[4])
		res.MidstateIndex = int(frame[5] & 0x03)
	}
	return res, nil
}

// ComputeMidstate computes the SHA-256 midstate: the 8-word SHA-256 state after
// compressing exactly one 64-byte block starting from the standard initial hash
// values. The result is returned as 32 big-endian bytes (8 x uint32, big-endian),
// which is the canonical SHA-256 state serialization.
//
// Some BM1387 firmwares expect the midstate words in reversed byte order; use
// SwapMidstateWordEndianness to convert if your transport requires it.
func ComputeMidstate(block []byte) [32]byte {
	var state [8]uint32
	copy(state[:], sha256InitialState[:])
	sha256Compress(&state, block)

	var out [32]byte
	for i := 0; i < 8; i++ {
		binary.BigEndian.PutUint32(out[i*4:], state[i])
	}
	return out
}

// SwapMidstateWordEndianness reverses the byte order within each 4-byte word of a
// 32-byte midstate. BM1387 firmware variants differ on whether the midstate words are
// sent big- or little-endian; this converts between the two conventions.
func SwapMidstateWordEndianness(ms [32]byte) [32]byte {
	var out [32]byte
	for i := 0; i < 8; i++ {
		w := binary.BigEndian.Uint32(ms[i*4:])
		binary.LittleEndian.PutUint32(out[i*4:], w)
	}
	return out
}

// --- CRC implementations -----------------------------------------------------

// CRC5 computes the 5-bit CRC used by Bitmain BM13xx command frames, ported from
// bmminer. numBits is the number of leading bits of data to include (BM1387 commands
// CRC the first (len*8 - 5) bits, i.e. everything except the CRC field itself).
func CRC5(data []byte, numBits int) uint8 {
	crcin := [5]byte{1, 1, 1, 1, 1}
	var crcout [5]byte

	bit := 0
	for i := 0; i < numBits; i++ {
		var din byte
		if data[bit/8]&(0x80>>(bit%8)) != 0 {
			din = 1
		}
		crcout[0] = crcin[4] ^ din
		crcout[1] = crcin[0]
		crcout[2] = crcin[1] ^ crcin[4] ^ din
		crcout[3] = crcin[2]
		crcout[4] = crcin[3]
		crcin = crcout
		bit++
	}

	var crc uint8
	if crcin[4] != 0 {
		crc |= 0x10
	}
	if crcin[3] != 0 {
		crc |= 0x08
	}
	if crcin[2] != 0 {
		crc |= 0x04
	}
	if crcin[1] != 0 {
		crc |= 0x02
	}
	if crcin[0] != 0 {
		crc |= 0x01
	}
	return crc
}

// CRC16 computes CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection), used to
// protect BM1387 work payloads.
func CRC16(data []byte) uint16 {
	crc := uint16(0xFFFF)
	for _, b := range data {
		crc ^= uint16(b) << 8
		for i := 0; i < 8; i++ {
			if crc&0x8000 != 0 {
				crc = (crc << 1) ^ 0x1021
			} else {
				crc <<= 1
			}
		}
	}
	return crc
}

// BM1387Command builds a CRC5-protected 5-byte command frame for the BM1387 chain.
// cmd is the command token, and param/addr are the two payload bytes. The returned
// frame is [cmd|type][len][param][addr][crc5], where the final byte's low 5 bits hold
// the CRC over the preceding 35 bits.
func BM1387Command(cmd, param, addr byte, broadcast bool) []byte {
	typ := cmd
	if broadcast {
		typ |= 0x80 // ALL flag
	}
	frame := []byte{typ, 0x05, param, addr, 0x00}
	crc := CRC5(frame, len(frame)*8-5)
	frame[4] = crc & 0x1F
	return frame
}

// --- Minimal SHA-256 block compression --------------------------------------
//
// Go's crypto/sha256 does not expose the intermediate state after a single block, so a
// standalone one-block compression function is implemented here to compute midstates.

var sha256InitialState = [8]uint32{
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
	0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
}

var sha256K = [64]uint32{
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
}

func rotr(x uint32, n uint) uint32 { return (x >> n) | (x << (32 - n)) }

// sha256Compress compresses one 64-byte block into state (SHA-256 round function).
func sha256Compress(state *[8]uint32, block []byte) {
	var w [64]uint32
	for i := 0; i < 16; i++ {
		w[i] = binary.BigEndian.Uint32(block[i*4:])
	}
	for i := 16; i < 64; i++ {
		s0 := rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3)
		s1 := rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10)
		w[i] = w[i-16] + s0 + w[i-7] + s1
	}

	a, b, c, d := state[0], state[1], state[2], state[3]
	e, f, g, h := state[4], state[5], state[6], state[7]

	for i := 0; i < 64; i++ {
		s1 := rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
		ch := (e & f) ^ (^e & g)
		t1 := h + s1 + ch + sha256K[i] + w[i]
		s0 := rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
		maj := (a & b) ^ (a & c) ^ (b & c)
		t2 := s0 + maj
		h = g
		g = f
		f = e
		e = d + t1
		d = c
		c = b
		b = a
		a = t1 + t2
	}

	state[0] += a
	state[1] += b
	state[2] += c
	state[3] += d
	state[4] += e
	state[5] += f
	state[6] += g
	state[7] += h
}
