package hardware

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"strings"
	"time"

	"hasher/pkg/hashing/core"
)

// ConnectionType describes how the host reaches the ASIC chips.
type ConnectionType string

const (
	// ConnectionUSB is a direct USB bulk connection to the hashboard (Antminer S1/S2/S3).
	ConnectionUSB ConnectionType = "USB"
	// ConnectionNetwork is a TCP/IP connection to the miner's on-board controller
	// (BeagleBone/Zynq) running cgminer or bmminer (Antminer S5 and newer).
	ConnectionNetwork ConnectionType = "Network"
	// ConnectionUART is a direct serial chain to the BM13xx chips (advanced/direct-drive).
	ConnectionUART ConnectionType = "UART"
)

// ProtocolFamily identifies the wire protocol used to push work and read nonces.
type ProtocolFamily string

const (
	// ProtocolBM1382USB is the original Bitmain USB bulk protocol used by HASHER for
	// the S1/S2/S3 (BM1380/BM1382). Work is pushed as a full 80-byte header via the
	// 0x52 TXTASK token (see internal/driver/device/controller.go).
	ProtocolBM1382USB ProtocolFamily = "bm1382-usb"
	// ProtocolBM1387Chain is the BM1387 serial chain protocol used by the S9 family and
	// the T9+. Work is pushed as a SHA-256 midstate plus a 12-byte header tail rather
	// than a full header. See bm1387_header.go for the encoder.
	ProtocolBM1387Chain ProtocolFamily = "bm1387-chain"
	// ProtocolCGMinerAPI drives the miner through a running cgminer/bmminer instance over
	// its JSON-RPC API (default port 4028). The chip-level protocol is handled by the
	// miner firmware, so this path is model-agnostic for S5 and newer control-board rigs.
	ProtocolCGMinerAPI ProtocolFamily = "cgminer-api"
)

// MinerProfile captures the fixed hardware characteristics of a supported ASIC miner.
// These are nominal factory specifications; real chip counts and hash rates vary by
// batch, firmware, and tuning. The values drive capability reporting, work generation,
// and nonce interpretation across the different miner generations.
type MinerProfile struct {
	// Model is the canonical, human-readable model name.
	Model string `json:"model"`
	// Aliases are lower-case match strings looked for inside the miner's reported
	// type string (from cgminer/bmminer) or an ASIC_MODEL override. Order the registry
	// most-specific-first so, e.g., "s19 pro" wins over "s19".
	Aliases []string `json:"aliases"`
	// Chip is the BM13xx ASIC used by this model.
	Chip string `json:"chip"`
	// Process is the silicon process node (informational).
	Process string `json:"process"`
	// ChipCount is the total number of hashing chips across all boards.
	ChipCount int `json:"chip_count"`
	// BoardCount is the number of hashboards.
	BoardCount int `json:"board_count"`
	// NominalHashRate is the factory-rated hash rate in hashes per second.
	NominalHashRate uint64 `json:"nominal_hash_rate"`
	// DefaultFrequencyMHz is the stock chip clock.
	DefaultFrequencyMHz int `json:"default_frequency_mhz"`
	// Connection is how the host reaches the hardware.
	Connection ConnectionType `json:"connection"`
	// Protocol is the wire protocol used to push work.
	Protocol ProtocolFamily `json:"protocol"`
	// DefaultDevicePath is the kernel device node for USB-connected models.
	DefaultDevicePath string `json:"default_device_path,omitempty"`
	// APIPort is the cgminer/bmminer JSON-RPC port for network models.
	APIPort int `json:"api_port,omitempty"`
	// VersionRolling indicates AsicBoost / version-rolling support.
	VersionRolling bool `json:"version_rolling"`
	// Notes flags approximations or caveats for a model.
	Notes string `json:"notes,omitempty"`
}

// th converts terahashes/second to hashes/second.
func th(x float64) uint64 { return uint64(x * 1e12) }

// gh converts gigahashes/second to hashes/second.
func gh(x float64) uint64 { return uint64(x * 1e9) }

// minerRegistry lists every supported miner, ordered most-specific-first so that
// substring alias matching resolves the narrowest model (e.g. "S19 Pro" before "S19",
// "S9j" before "S9"). New models should be inserted in the correct specificity slot.
var minerRegistry = []MinerProfile{
	// --- 5nm / 3nm flagship generation -----------------------------------------
	{
		Model: "Antminer S21", Aliases: []string{"antminer s21", "s21"},
		Chip: "BM1370", Process: "5nm", ChipCount: 0, BoardCount: 3,
		NominalHashRate: th(200), DefaultFrequencyMHz: 0,
		Connection: ConnectionNetwork, Protocol: ProtocolCGMinerAPI, APIPort: 4028,
		VersionRolling: true, Notes: "Chip count varies by sub-model; hash rate nominal.",
	},
	{
		Model: "Antminer S19 XP", Aliases: []string{"antminer s19 xp", "s19xp", "s19 xp"},
		Chip: "BM1366", Process: "5nm", ChipCount: 0, BoardCount: 3,
		NominalHashRate: th(140), DefaultFrequencyMHz: 0,
		Connection: ConnectionNetwork, Protocol: ProtocolCGMinerAPI, APIPort: 4028,
		VersionRolling: true, Notes: "Chip count varies by sub-model; hash rate nominal.",
	},
	// --- 7nm generation --------------------------------------------------------
	{
		Model: "Antminer S19j Pro", Aliases: []string{"antminer s19j pro", "s19j pro", "s19jpro", "s19j"},
		Chip: "BM1398", Process: "7nm", ChipCount: 0, BoardCount: 3,
		NominalHashRate: th(104), DefaultFrequencyMHz: 0,
		Connection: ConnectionNetwork, Protocol: ProtocolCGMinerAPI, APIPort: 4028,
		VersionRolling: true, Notes: "Chip count varies by sub-model; hash rate nominal.",
	},
	{
		Model: "Antminer S19 Pro", Aliases: []string{"antminer s19 pro", "s19 pro", "s19pro"},
		Chip: "BM1398", Process: "7nm", ChipCount: 0, BoardCount: 3,
		NominalHashRate: th(110), DefaultFrequencyMHz: 0,
		Connection: ConnectionNetwork, Protocol: ProtocolCGMinerAPI, APIPort: 4028,
		VersionRolling: true, Notes: "Chip count varies by sub-model; hash rate nominal.",
	},
	{
		Model: "Antminer S19", Aliases: []string{"antminer s19", "s19"},
		Chip: "BM1398", Process: "7nm", ChipCount: 0, BoardCount: 3,
		NominalHashRate: th(95), DefaultFrequencyMHz: 0,
		Connection: ConnectionNetwork, Protocol: ProtocolCGMinerAPI, APIPort: 4028,
		VersionRolling: true, Notes: "Chip count varies by sub-model; hash rate nominal.",
	},
	{
		Model: "Antminer T17", Aliases: []string{"antminer t17", "t17"},
		Chip: "BM1397", Process: "7nm", ChipCount: 0, BoardCount: 3,
		NominalHashRate: th(40), DefaultFrequencyMHz: 0,
		Connection: ConnectionNetwork, Protocol: ProtocolCGMinerAPI, APIPort: 4028,
		VersionRolling: true, Notes: "Chip count varies by sub-model; hash rate nominal.",
	},
	{
		Model: "Antminer S17", Aliases: []string{"antminer s17", "s17"},
		Chip: "BM1397", Process: "7nm", ChipCount: 0, BoardCount: 3,
		NominalHashRate: th(56), DefaultFrequencyMHz: 0,
		Connection: ConnectionNetwork, Protocol: ProtocolCGMinerAPI, APIPort: 4028,
		VersionRolling: true, Notes: "Chip count varies by sub-model; hash rate nominal.",
	},
	{
		Model: "Antminer T15", Aliases: []string{"antminer t15", "t15"},
		Chip: "BM1391", Process: "7nm", ChipCount: 0, BoardCount: 2,
		NominalHashRate: th(23), DefaultFrequencyMHz: 0,
		Connection: ConnectionNetwork, Protocol: ProtocolCGMinerAPI, APIPort: 4028,
		VersionRolling: true, Notes: "Chip count varies by sub-model; hash rate nominal.",
	},
	{
		Model: "Antminer S15", Aliases: []string{"antminer s15", "s15"},
		Chip: "BM1391", Process: "7nm", ChipCount: 0, BoardCount: 2,
		NominalHashRate: th(28), DefaultFrequencyMHz: 0,
		Connection: ConnectionNetwork, Protocol: ProtocolCGMinerAPI, APIPort: 4028,
		VersionRolling: true, Notes: "Chip count varies by sub-model; hash rate nominal.",
	},
	// --- 16nm BM1387 generation (the classic "newer" miners) -------------------
	{
		Model: "Antminer T9+", Aliases: []string{"antminer t9+", "t9+", "t9plus"},
		Chip: "BM1387", Process: "16nm", ChipCount: 162, BoardCount: 3,
		NominalHashRate: th(10.5), DefaultFrequencyMHz: 600,
		Connection: ConnectionNetwork, Protocol: ProtocolCGMinerAPI, APIPort: 4028,
		VersionRolling: false, Notes: "162 chips (3 boards x 54).",
	},
	{
		Model: "Antminer S9k", Aliases: []string{"antminer s9k", "s9k"},
		Chip: "BM1387", Process: "16nm", ChipCount: 0, BoardCount: 3,
		NominalHashRate: th(13.5), DefaultFrequencyMHz: 650,
		Connection: ConnectionNetwork, Protocol: ProtocolCGMinerAPI, APIPort: 4028,
		VersionRolling: false,
	},
	{
		Model: "Antminer S9 SE", Aliases: []string{"antminer s9 se", "s9se", "s9 se"},
		Chip: "BM1387", Process: "16nm", ChipCount: 0, BoardCount: 3,
		NominalHashRate: th(16), DefaultFrequencyMHz: 650,
		Connection: ConnectionNetwork, Protocol: ProtocolCGMinerAPI, APIPort: 4028,
		VersionRolling: false,
	},
	{
		Model: "Antminer S9j", Aliases: []string{"antminer s9j", "s9j"},
		Chip: "BM1387", Process: "16nm", ChipCount: 189, BoardCount: 3,
		NominalHashRate: th(14.5), DefaultFrequencyMHz: 650,
		Connection: ConnectionNetwork, Protocol: ProtocolCGMinerAPI, APIPort: 4028,
		VersionRolling: false, Notes: "189 chips (3 boards x 63).",
	},
	{
		Model: "Antminer S9i", Aliases: []string{"antminer s9i", "s9i"},
		Chip: "BM1387", Process: "16nm", ChipCount: 189, BoardCount: 3,
		NominalHashRate: th(14), DefaultFrequencyMHz: 650,
		Connection: ConnectionNetwork, Protocol: ProtocolCGMinerAPI, APIPort: 4028,
		VersionRolling: false, Notes: "189 chips (3 boards x 63).",
	},
	{
		// Base S9 also catches a bare "bm1387" chip hint.
		Model: "Antminer S9", Aliases: []string{"antminer s9", "s9", "bm1387"},
		Chip: "BM1387", Process: "16nm", ChipCount: 189, BoardCount: 3,
		NominalHashRate: th(13.5), DefaultFrequencyMHz: 650,
		Connection: ConnectionNetwork, Protocol: ProtocolCGMinerAPI, APIPort: 4028,
		VersionRolling: false, Notes: "189 chips (3 boards x 63). Direct-drive uses the BM1387 chain protocol.",
	},
	// --- Earlier control-board generations -------------------------------------
	{
		Model: "Antminer S7", Aliases: []string{"antminer s7", "s7", "bm1385"},
		Chip: "BM1385", Process: "28nm", ChipCount: 162, BoardCount: 3,
		NominalHashRate: th(4.73), DefaultFrequencyMHz: 600,
		Connection: ConnectionNetwork, Protocol: ProtocolCGMinerAPI, APIPort: 4028,
		VersionRolling: false, Notes: "162 chips (3 boards x 54).",
	},
	{
		Model: "Antminer S5", Aliases: []string{"antminer s5", "s5", "bm1384"},
		Chip: "BM1384", Process: "28nm", ChipCount: 60, BoardCount: 1,
		NominalHashRate: gh(1155), DefaultFrequencyMHz: 350,
		Connection: ConnectionNetwork, Protocol: ProtocolCGMinerAPI, APIPort: 4028,
		VersionRolling: false,
	},
	// --- Original USB BM1382 generation (HASHER native target) -----------------
	{
		Model: "Antminer S3", Aliases: []string{"antminer s3", "s3", "bm1382"},
		Chip: "BM1382", Process: "28nm", ChipCount: 32, BoardCount: 2,
		NominalHashRate: gh(478), DefaultFrequencyMHz: 250,
		Connection: ConnectionUSB, Protocol: ProtocolBM1382USB,
		DefaultDevicePath: "/dev/bitmain-asic", APIPort: 4028,
		VersionRolling: false, Notes: "HASHER's original direct-USB target. 32 chips (2 boards x 16).",
	},
	{
		Model: "Antminer S2", Aliases: []string{"antminer s2", "s2"},
		Chip: "BM1382", Process: "28nm", ChipCount: 32, BoardCount: 2,
		NominalHashRate: th(1), DefaultFrequencyMHz: 250,
		Connection: ConnectionUSB, Protocol: ProtocolBM1382USB,
		DefaultDevicePath: "/dev/bitmain-asic", APIPort: 4028,
		VersionRolling: false, Notes: "HASHER's original direct-USB target.",
	},
	{
		Model: "Antminer S1", Aliases: []string{"antminer s1", "s1", "bm1380"},
		Chip: "BM1380", Process: "55nm", ChipCount: 32, BoardCount: 2,
		NominalHashRate: gh(180), DefaultFrequencyMHz: 350,
		Connection: ConnectionUSB, Protocol: ProtocolBM1382USB,
		DefaultDevicePath: "/dev/bitmain-asic", APIPort: 4028,
		VersionRolling: false,
	},
}

// DefaultProfileModel is the model assumed when no model can be detected. It preserves
// HASHER's original behaviour of targeting the Antminer S3 (BM1382) over USB.
const DefaultProfileModel = "Antminer S3"

// AllProfiles returns a copy of the supported miner registry, most-specific-first.
func AllProfiles() []MinerProfile {
	out := make([]MinerProfile, len(minerRegistry))
	copy(out, minerRegistry)
	return out
}

// ProfileByModel returns the profile whose canonical model name matches name
// (case-insensitive, exact).
func ProfileByModel(name string) (MinerProfile, bool) {
	name = strings.TrimSpace(strings.ToLower(name))
	for _, p := range minerRegistry {
		if strings.ToLower(p.Model) == name {
			return p, true
		}
	}
	return MinerProfile{}, false
}

// DefaultProfile returns the fallback profile (Antminer S3).
func DefaultProfile() MinerProfile {
	p, ok := ProfileByModel(DefaultProfileModel)
	if !ok {
		// Should never happen; registry always contains the default.
		return minerRegistry[len(minerRegistry)-1]
	}
	return p
}

// DetectProfile resolves a free-form hint (a model name, a cgminer "Type" string, a
// chip name, or an ASIC_MODEL override) to a known miner profile. Matching is
// case-insensitive substring matching against each profile's aliases, in registry
// order (most-specific-first), so "Antminer S19 Pro" resolves to the S19 Pro and not
// the base S19. Returns false if nothing matches.
func DetectProfile(hint string) (MinerProfile, bool) {
	h := strings.ToLower(strings.TrimSpace(hint))
	if h == "" {
		return MinerProfile{}, false
	}
	for _, p := range minerRegistry {
		for _, a := range p.Aliases {
			if a != "" && strings.Contains(h, a) {
				return p, true
			}
		}
	}
	return MinerProfile{}, false
}

// HardwareInfo builds a core.HardwareInfo description from the profile.
func (p MinerProfile) HardwareInfo() *core.HardwareInfo {
	devicePath := p.DefaultDevicePath
	if devicePath == "" {
		devicePath = fmt.Sprintf("cgminer-api:%d", p.APIPort)
	}
	return &core.HardwareInfo{
		DevicePath:     devicePath,
		ChipCount:      p.ChipCount,
		Version:        p.Chip,
		ConnectionType: string(p.Connection),
		Metadata: map[string]string{
			"model":           p.Model,
			"chip":            p.Chip,
			"process":         p.Process,
			"protocol":        string(p.Protocol),
			"board_count":     fmt.Sprintf("%d", p.BoardCount),
			"frequency_mhz":   fmt.Sprintf("%d", p.DefaultFrequencyMHz),
			"version_rolling": fmt.Sprintf("%t", p.VersionRolling),
		},
	}
}

// Capabilities builds a core.Capabilities description from the profile. available
// controls IsHardware/ProductionReady so callers can report a known-but-absent miner.
func (p MinerProfile) Capabilities(available bool) *core.Capabilities {
	maxBatch := p.ChipCount
	if maxBatch <= 0 {
		maxBatch = 256
	}
	return &core.Capabilities{
		Name:            fmt.Sprintf("ASIC Hardware (%s)", p.Model),
		IsHardware:      available,
		HashRate:        p.NominalHashRate,
		ProductionReady: available,
		MaxBatchSize:    maxBatch,
		AvgLatencyUs:    100,
		HardwareInfo:    p.HardwareInfo(),
	}
}

// --- Model detection helpers -------------------------------------------------

// DetectModelHint returns the best available model hint for the current host, in
// priority order:
//  1. The ASIC_MODEL environment variable (explicit override).
//  2. The miner "Type" reported by a reachable cgminer/bmminer API (CGMINER_HOST or
//     127.0.0.1:4028).
//
// It returns an empty string if nothing is discoverable, in which case callers should
// fall back to DefaultProfile.
func DetectModelHint() string {
	if m := strings.TrimSpace(os.Getenv("ASIC_MODEL")); m != "" {
		return m
	}

	host := strings.TrimSpace(os.Getenv("CGMINER_HOST"))
	if host == "" {
		host = "127.0.0.1"
	}
	if t, err := queryCGMinerType(host, 4028, 2*time.Second); err == nil && t != "" {
		return t
	}
	return ""
}

// queryCGMinerType asks a cgminer/bmminer instance for its hardware type via the
// JSON-RPC API. It tries the "stats" command (which carries a "Type" field on Bitmain
// firmware) and falls back to "version". This is a self-contained client so the
// hardware package stays independent of the USB/eBPF driver packages.
func queryCGMinerType(host string, port int, timeout time.Duration) (string, error) {
	for _, cmd := range []string{"stats", "version"} {
		resp, err := cgminerCommand(host, port, cmd, timeout)
		if err != nil {
			continue
		}
		if t := extractType(resp); t != "" {
			return t, nil
		}
	}
	return "", fmt.Errorf("no type reported by cgminer at %s:%d", host, port)
}

// extractType walks a decoded cgminer response looking for a Bitmain "Type" string.
func extractType(resp map[string]interface{}) string {
	// Common shapes: {"STATS":[{"Type":"Antminer S9"}...]} or {"VERSION":[{"Type":"..."}]}.
	for _, key := range []string{"STATS", "VERSION", "DEVS", "SUMMARY"} {
		arr, ok := resp[key].([]interface{})
		if !ok {
			continue
		}
		for _, item := range arr {
			m, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			for _, field := range []string{"Type", "Model", "Name"} {
				if v, ok := m[field].(string); ok && v != "" {
					return v
				}
			}
		}
	}
	return ""
}

// cgminerCommand sends a single JSON-RPC command to a cgminer/bmminer instance and
// returns the decoded response.
func cgminerCommand(host string, port int, command string, timeout time.Duration) (map[string]interface{}, error) {
	addr := net.JoinHostPort(host, fmt.Sprintf("%d", port))
	conn, err := net.DialTimeout("tcp", addr, timeout)
	if err != nil {
		return nil, err
	}
	defer conn.Close()
	_ = conn.SetDeadline(time.Now().Add(timeout))

	payload, err := json.Marshal(map[string]string{"command": command})
	if err != nil {
		return nil, err
	}
	if _, err := conn.Write(append(payload, 0x00)); err != nil {
		return nil, err
	}

	var buf bytes.Buffer
	tmp := make([]byte, 4096)
	for {
		n, err := conn.Read(tmp)
		if n > 0 {
			buf.Write(tmp[:n])
		}
		if err != nil {
			break
		}
	}

	raw := bytes.ReplaceAll(buf.Bytes(), []byte{0x00}, nil)
	var out map[string]interface{}
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, err
	}
	return out, nil
}
