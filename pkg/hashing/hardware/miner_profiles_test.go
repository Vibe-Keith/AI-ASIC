package hardware

import "testing"

func TestDetectProfileResolvesModels(t *testing.T) {
	cases := []struct {
		hint      string
		wantModel string
		wantChip  string
	}{
		{"Antminer S9", "Antminer S9", "BM1387"},
		{"antminer s9i", "Antminer S9i", "BM1387"},
		{"S9j", "Antminer S9j", "BM1387"},
		{"Antminer T9+", "Antminer T9+", "BM1387"},
		{"Antminer S19 Pro", "Antminer S19 Pro", "BM1398"},
		{"antminer s19j pro", "Antminer S19j Pro", "BM1398"},
		{"Antminer S19", "Antminer S19", "BM1398"},
		{"Antminer S17", "Antminer S17", "BM1397"},
		{"Antminer S3", "Antminer S3", "BM1382"},
		{"bm1387", "Antminer S9", "BM1387"},
		{"Some Antminer S21 Hyd", "Antminer S21", "BM1370"},
	}

	for _, c := range cases {
		p, ok := DetectProfile(c.hint)
		if !ok {
			t.Errorf("DetectProfile(%q): expected a match, got none", c.hint)
			continue
		}
		if p.Model != c.wantModel {
			t.Errorf("DetectProfile(%q): model = %q, want %q", c.hint, p.Model, c.wantModel)
		}
		if p.Chip != c.wantChip {
			t.Errorf("DetectProfile(%q): chip = %q, want %q", c.hint, p.Chip, c.wantChip)
		}
	}
}

func TestDetectProfileSpecificityS19NotS9(t *testing.T) {
	p, ok := DetectProfile("Antminer S19")
	if !ok || p.Model != "Antminer S19" {
		t.Fatalf("S19 should not resolve to S9; got %q (ok=%v)", p.Model, ok)
	}
}

func TestDetectProfileUnknown(t *testing.T) {
	if _, ok := DetectProfile("Whatsminer M30S"); ok {
		t.Error("expected no match for an unsupported miner")
	}
	if _, ok := DetectProfile(""); ok {
		t.Error("expected no match for empty hint")
	}
}

func TestDefaultProfileIsS3(t *testing.T) {
	p := DefaultProfile()
	if p.Model != "Antminer S3" {
		t.Errorf("default profile = %q, want Antminer S3", p.Model)
	}
	if p.Protocol != ProtocolBM1382USB {
		t.Errorf("default protocol = %q, want %q", p.Protocol, ProtocolBM1382USB)
	}
	if p.DefaultDevicePath != "/dev/bitmain-asic" {
		t.Errorf("default device path = %q", p.DefaultDevicePath)
	}
}

func TestProfileCapabilities(t *testing.T) {
	p, _ := DetectProfile("Antminer S9")

	avail := p.Capabilities(true)
	if !avail.IsHardware || !avail.ProductionReady {
		t.Error("available S9 should report hardware + production ready")
	}
	if avail.HashRate != th(13.5) {
		t.Errorf("S9 hash rate = %d, want %d", avail.HashRate, th(13.5))
	}
	if avail.HardwareInfo.ChipCount != 189 {
		t.Errorf("S9 chip count = %d, want 189", avail.HardwareInfo.ChipCount)
	}
	if avail.HardwareInfo.ConnectionType != string(ConnectionNetwork) {
		t.Errorf("S9 connection = %q, want Network", avail.HardwareInfo.ConnectionType)
	}
	if got := avail.HardwareInfo.Metadata["protocol"]; got != string(ProtocolCGMinerAPI) {
		t.Errorf("S9 protocol metadata = %q, want %q", got, ProtocolCGMinerAPI)
	}

	unavail := p.Capabilities(false)
	if unavail.IsHardware || unavail.ProductionReady {
		t.Error("unavailable profile should not report hardware/production ready")
	}
}

func TestRegistryIntegrity(t *testing.T) {
	seen := map[string]bool{}
	for _, p := range AllProfiles() {
		if p.Model == "" {
			t.Error("profile with empty model")
		}
		if seen[p.Model] {
			t.Errorf("duplicate model in registry: %q", p.Model)
		}
		seen[p.Model] = true
		if len(p.Aliases) == 0 {
			t.Errorf("%s: no aliases", p.Model)
		}
		if p.Chip == "" {
			t.Errorf("%s: no chip", p.Model)
		}
		if p.Protocol == "" {
			t.Errorf("%s: no protocol", p.Model)
		}
	}
}

func TestExtractType(t *testing.T) {
	resp := map[string]interface{}{
		"STATS": []interface{}{
			map[string]interface{}{"Type": "Antminer S9"},
		},
	}
	if got := extractType(resp); got != "Antminer S9" {
		t.Errorf("extractType = %q, want Antminer S9", got)
	}

	if got := extractType(map[string]interface{}{}); got != "" {
		t.Errorf("extractType on empty = %q, want empty", got)
	}
}
