package hardware

import (
	"bytes"
	"encoding/hex"
	"testing"
)

// TestSHA256CompressEmptyString validates the standalone block-compression function by
// hashing the padded block for the empty string and comparing against SHA-256("") =
// e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855. This proves
// ComputeMidstate is correct, since midstate reuses the same compression round.
func TestSHA256CompressEmptyString(t *testing.T) {
	block := make([]byte, 64)
	block[0] = 0x80 // single padding bit; message length is 0

	ms := ComputeMidstate(block)
	got := hex.EncodeToString(ms[:])
	want := "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
	if got != want {
		t.Fatalf("SHA-256 compression mismatch:\n got  %s\n want %s", got, want)
	}
}

func TestComputeMidstateDeterministic(t *testing.T) {
	block := make([]byte, 64)
	for i := range block {
		block[i] = byte(i)
	}
	a := ComputeMidstate(block)
	b := ComputeMidstate(block)
	if a != b {
		t.Error("ComputeMidstate is not deterministic")
	}
}

func TestSwapMidstateWordEndianness(t *testing.T) {
	var ms [32]byte
	for i := range ms {
		ms[i] = byte(i)
	}
	swapped := SwapMidstateWordEndianness(ms)
	// First word bytes 00 01 02 03 -> 03 02 01 00.
	if swapped[0] != 0x03 || swapped[1] != 0x02 || swapped[2] != 0x01 || swapped[3] != 0x00 {
		t.Errorf("word swap wrong: %x", swapped[0:4])
	}
	// Double swap restores the original.
	if SwapMidstateWordEndianness(swapped) != ms {
		t.Error("double swap should restore original")
	}
}

func TestNewBM1387WorkFromHeader(t *testing.T) {
	header := make([]byte, 80)
	for i := range header {
		header[i] = byte(i)
	}

	w, err := NewBM1387WorkFromHeader(header, 0xDEADBEEF)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(w.Midstates) != 1 {
		t.Fatalf("expected 1 midstate, got %d", len(w.Midstates))
	}
	// Data must equal header bytes 64..75 (merkle tail + ntime + nbits), not the nonce.
	if !bytes.Equal(w.Data[:], header[64:76]) {
		t.Errorf("work data = %x, want %x", w.Data[:], header[64:76])
	}
	if w.Midstates[0] != ComputeMidstate(header[0:64]) {
		t.Error("midstate does not match first 64 header bytes")
	}
}

func TestNewBM1387WorkFromHeaderBadLength(t *testing.T) {
	if _, err := NewBM1387WorkFromHeader(make([]byte, 79), 0); err == nil {
		t.Error("expected error for non-80-byte header")
	}
}

func TestNewBM1387WorkAsicBoost(t *testing.T) {
	header := make([]byte, 80)
	versions := []uint32{0x20000000, 0x20000004, 0x20000008, 0x2000000c}

	w, err := NewBM1387WorkAsicBoost(header, 1, versions)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(w.Midstates) != 4 {
		t.Fatalf("expected 4 midstates, got %d", len(w.Midstates))
	}
	// Different versions must yield different midstates.
	if w.Midstates[0] == w.Midstates[1] {
		t.Error("distinct versions produced identical midstates")
	}

	if _, err := NewBM1387WorkAsicBoost(header, 1, nil); err == nil {
		t.Error("expected error for zero versions")
	}
	if _, err := NewBM1387WorkAsicBoost(header, 1, make([]uint32, 5)); err == nil {
		t.Error("expected error for >4 versions")
	}
}

func TestBM1387WorkEncodeLayout(t *testing.T) {
	header := make([]byte, 80)
	w, _ := NewBM1387WorkFromHeader(header, 0x01020304)

	frame, err := w.Encode()
	if err != nil {
		t.Fatalf("encode error: %v", err)
	}
	// 4 (work id) + 32 (midstate) + 12 (data) + 2 (crc) = 50 bytes.
	if len(frame) != 50 {
		t.Fatalf("frame length = %d, want 50", len(frame))
	}
	// Work ID little-endian at the front.
	if frame[0] != 0x04 || frame[1] != 0x03 || frame[2] != 0x02 || frame[3] != 0x01 {
		t.Errorf("work id encoding wrong: %x", frame[0:4])
	}
	// CRC16 over everything but the trailing 2 bytes.
	wantCRC := CRC16(frame[:len(frame)-2])
	gotCRC := uint16(frame[len(frame)-2])<<8 | uint16(frame[len(frame)-1])
	if gotCRC != wantCRC {
		t.Errorf("trailing crc = %04x, want %04x", gotCRC, wantCRC)
	}
}

func TestBM1387WorkEncodeAsicBoostLength(t *testing.T) {
	header := make([]byte, 80)
	w, _ := NewBM1387WorkAsicBoost(header, 1, []uint32{1, 2, 3, 4})
	frame, _ := w.Encode()
	// 4 + 4*32 + 12 + 2 = 146.
	if len(frame) != 146 {
		t.Fatalf("asicboost frame length = %d, want 146", len(frame))
	}
}

func TestCRC16KnownVector(t *testing.T) {
	// CRC-16/CCITT-FALSE("123456789") = 0x29B1.
	if got := CRC16([]byte("123456789")); got != 0x29B1 {
		t.Errorf("CRC16 = %04x, want 29B1", got)
	}
}

func TestCRC5Deterministic(t *testing.T) {
	data := []byte{0x05, 0x00, 0x00, 0x00, 0x00}
	a := CRC5(data, len(data)*8-5)
	b := CRC5(data, len(data)*8-5)
	if a != b {
		t.Error("CRC5 not deterministic")
	}
	if a > 0x1F {
		t.Errorf("CRC5 must be 5 bits, got %02x", a)
	}
}

func TestBM1387Command(t *testing.T) {
	frame := BM1387Command(BM1387CmdChainInactive, 0x00, 0x00, true)
	if len(frame) != 5 {
		t.Fatalf("command frame length = %d, want 5", len(frame))
	}
	if frame[0]&0x80 == 0 {
		t.Error("broadcast command should set the ALL (0x80) flag")
	}
	if frame[4] > 0x1F {
		t.Errorf("crc5 field must be 5 bits, got %02x", frame[4])
	}
}

func TestParseNonceResponse(t *testing.T) {
	// nonce 0x11223344 big-endian, work id 0x07, midstate index 2.
	frame := []byte{0x11, 0x22, 0x33, 0x44, 0x07, 0x02, 0x00}
	res, err := ParseNonceResponse(frame)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Nonce != 0x11223344 {
		t.Errorf("nonce = %08x, want 11223344", res.Nonce)
	}
	if res.WorkID != 0x07 {
		t.Errorf("work id = %d, want 7", res.WorkID)
	}
	if res.MidstateIndex != 2 {
		t.Errorf("midstate index = %d, want 2", res.MidstateIndex)
	}

	if _, err := ParseNonceResponse([]byte{0x00}); err == nil {
		t.Error("expected error for short frame")
	}
}
