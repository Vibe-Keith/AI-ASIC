package hardware

import (
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"time"

	"hasher/pkg/hashing/core"
)

// DeviceDetector performs hardware detection for available hashing methods
type DeviceDetector struct {
	detectedMethods map[string]bool
	capabilities    map[string]*core.Capabilities
}

// NewDeviceDetector creates a new hardware detector
func NewDeviceDetector() *DeviceDetector {
	return &DeviceDetector{
		detectedMethods: make(map[string]bool),
		capabilities:    make(map[string]*core.Capabilities),
	}
}

// DetectAvailableMethods performs comprehensive hardware detection
func (d *DeviceDetector) DetectAvailableMethods() map[string]bool {
	d.detectASIC()
	d.detectCUDA()
	d.detectuBPF()
	d.detectSoftware() // Always available

	return d.detectedMethods
}

// detectASIC checks for ASIC device availability across all supported miner models.
//
// It resolves the connected miner from (in priority order) the ASIC_MODEL override, a
// reachable cgminer/bmminer API, or the presence of the direct-USB device node, then
// reports capabilities (chip type, chip count, hash rate, connection, protocol) from
// that model's profile. This replaces the previous hard-coded Antminer S3 (BM1382)
// assumption so newer miners such as the S9 (BM1387) and later are detected correctly.
func (d *DeviceDetector) detectASIC() {
	hint := DetectModelHint()
	profile, matched := DetectProfile(hint)
	if !matched {
		profile = DefaultProfile()
	}

	detectedBy := "profile_default"
	if matched && strings.TrimSpace(hint) != "" {
		detectedBy = fmt.Sprintf("model_hint:%q", hint)
	}

	// Decide availability based on the profile's connection type.
	var available bool
	var reason string
	switch profile.Connection {
	case ConnectionUSB:
		available, reason = d.probeUSBDevice(profile.DefaultDevicePath)
	default:
		// Network / control-board miners: reachable cgminer API means the hardware is up.
		available = cgminerReachable()
		if !available {
			// Fall back to the USB node in case a direct-drive setup is in use.
			if usbOK, _ := d.probeUSBDevice(profile.DefaultDevicePath); usbOK {
				available = true
			} else {
				reason = "cgminer/bmminer API not reachable (is the miner powered and on the network?)"
			}
		}
	}

	d.detectedMethods["asic"] = available
	caps := profile.Capabilities(available)
	if !available {
		caps.HashRate = 0
		caps.Reason = reason
	}
	if caps.HardwareInfo != nil {
		caps.HardwareInfo.Metadata["detected_by"] = detectedBy
	}
	d.capabilities["asic"] = caps
}

// probeUSBDevice checks whether a direct-USB ASIC device node exists and is openable.
func (d *DeviceDetector) probeUSBDevice(devicePath string) (bool, string) {
	if devicePath == "" {
		devicePath = "/dev/bitmain-asic"
	}
	if _, err := os.Stat(devicePath); err != nil {
		return false, fmt.Sprintf("Device not found: %v", err)
	}
	file, err := os.OpenFile(devicePath, os.O_RDONLY, 0)
	if err != nil {
		return false, fmt.Sprintf("Cannot access device: %v (is CGMiner running?)", err)
	}
	file.Close()
	return true, ""
}

// cgminerReachable reports whether a cgminer/bmminer JSON-RPC API is responding.
func cgminerReachable() bool {
	host := strings.TrimSpace(os.Getenv("CGMINER_HOST"))
	if host == "" {
		host = "127.0.0.1"
	}
	_, err := cgminerCommand(host, 4028, "version", 2*time.Second)
	return err == nil
}

// detectCUDA checks for CUDA GPU availability
func (d *DeviceDetector) detectCUDA() {
	// Check for nvidia-smi command
	cmd := exec.Command("nvidia-smi", "--query-gpu=name,compute_capability", "--format=csv,noheader,nounits")
	output, err := cmd.Output()
	if err != nil {
		d.detectedMethods["cuda"] = false
		d.capabilities["cuda"] = &core.Capabilities{
			Name:            "CUDA",
			IsHardware:      false,
			HashRate:        0,
			ProductionReady: false,
			Reason:          "nvidia-smi not found",
		}
		return
	}

	// Parse GPU information
	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	if len(lines) == 0 || strings.Contains(lines[0], "command not found") {
		d.detectedMethods["cuda"] = false
		d.capabilities["cuda"] = &core.Capabilities{
			Name:            "CUDA",
			IsHardware:      false,
			HashRate:        0,
			ProductionReady: false,
			Reason:          "No NVIDIA GPUs found",
		}
		return
	}

	// Parse first GPU
	gpuInfo := strings.Split(lines[0], ",")
	if len(gpuInfo) < 2 {
		d.detectedMethods["cuda"] = false
		d.capabilities["cuda"] = &core.Capabilities{
			Name:            "CUDA",
			IsHardware:      false,
			HashRate:        0,
			ProductionReady: false,
			Reason:          "Failed to parse GPU info",
		}
		return
	}

	gpuName := strings.TrimSpace(gpuInfo[0])
	computeCap := strings.TrimSpace(gpuInfo[1])

	d.detectedMethods["cuda"] = true
	d.capabilities["cuda"] = &core.Capabilities{
		Name:              "CUDA Simulator (Training Only)",
		IsHardware:        true,
		HashRate:          50000000000, // 50 GH/s
		ProductionReady:   false,       // Training only
		TrainingOptimized: true,
		MaxBatchSize:      1000,
		AvgLatencyUs:      50, // 50 microseconds
		HardwareInfo: &core.HardwareInfo{
			DevicePath:     "cuda",
			ChipCount:      1,
			Version:        fmt.Sprintf("CUDA %s", computeCap),
			ConnectionType: "PCIe",
			Metadata: map[string]string{
				"gpu_name":           gpuName,
				"compute_capability": computeCap,
				"detected_by":        "nvidia-smi",
			},
		},
	}
}

// detectuBPF checks for uBPF and CGMiner availability
func (d *DeviceDetector) detectuBPF() {
	cgminerPaths := []string{
		"/opt/cgminer/cgminer",
		"/usr/local/bin/cgminer",
		"/usr/bin/cgminer",
		"cgminer",
	}

	var cgminerPath string
	var cgminerAvailable bool

	// Search for CGMiner binary
	for _, path := range cgminerPaths {
		if _, err := os.Stat(path); err == nil {
			cgminerPath = path
			cgminerAvailable = true
			break
		}
	}

	// Check for API access
	apiAvailable := false
	if cgminerAvailable {
		client := &http.Client{Timeout: 2 * time.Second}
		resp, err := client.Get("http://127.0.0.1:4028/summary")
		if err == nil && resp.StatusCode == 200 {
			apiAvailable = true
			resp.Body.Close()
		}
	}

	// Check USB device fallback
	usbAvailable := false
	if _, err := os.Stat("/dev/bitmain-asic"); err == nil {
		usbAvailable = true
	}

	d.detectedMethods["ubpf"] = cgminerAvailable || apiAvailable || usbAvailable
	d.capabilities["ubpf"] = &core.Capabilities{
		Name:              "uBPF Simulator (USB + CGMiner)",
		IsHardware:        cgminerAvailable || apiAvailable,
		HashRate:          100000000, // 100 MH/s simulated
		ProductionReady:   false,     // Simulation only
		TrainingOptimized: false,
		MaxBatchSize:      50,
		AvgLatencyUs:      500, // 500 microseconds
		HardwareInfo: &core.HardwareInfo{
			DevicePath:     cgminerPath,
			ChipCount:      32, // Simulated
			Version:        "ubpf-sim",
			ConnectionType: "USB/API",
			Metadata: map[string]string{
				"cgminer_available": fmt.Sprintf("%t", cgminerAvailable),
				"api_available":     fmt.Sprintf("%t", apiAvailable),
				"usb_fallback":      fmt.Sprintf("%t", usbAvailable),
				"detected_by":       "binary_and_api_check",
			},
		},
	}
}

// detectSoftware always returns true for software method
func (d *DeviceDetector) detectSoftware() {
	d.detectedMethods["software"] = true
	d.capabilities["software"] = &core.Capabilities{
		Name:              "Software Fallback",
		IsHardware:        false,
		HashRate:          1000000, // 1 MH/s typical CPU
		ProductionReady:   true,
		TrainingOptimized: false,
		MaxBatchSize:      100,
		AvgLatencyUs:      1000, // 1 millisecond
		HardwareInfo: &core.HardwareInfo{
			DevicePath:     "software",
			ChipCount:      0,
			Version:        fmt.Sprintf("Go %s", runtime.Version()),
			ConnectionType: "none",
			Metadata: map[string]string{
				"os":          runtime.GOOS,
				"arch":        runtime.GOARCH,
				"detected_by": "runtime_detection",
			},
		},
	}
}

// GetCapabilities returns capabilities for a specific method
func (d *DeviceDetector) GetCapabilities(method string) *core.Capabilities {
	if caps, exists := d.capabilities[method]; exists {
		return caps
	}
	return &core.Capabilities{
		Name:            method,
		IsHardware:      false,
		ProductionReady: false,
		Reason:          "Unknown method",
	}
}

// GetAllCapabilities returns all detected capabilities
func (d *DeviceDetector) GetAllCapabilities() map[string]*core.Capabilities {
	result := make(map[string]*core.Capabilities)
	for method, caps := range d.capabilities {
		result[method] = caps
	}
	return result
}

// GetDetectionSummary returns a human-readable summary
func (d *DeviceDetector) GetDetectionSummary() string {
	var builder strings.Builder

	builder.WriteString("Hardware Detection Summary:\n")
	builder.WriteString("========================\n\n")

	for method, available := range d.detectedMethods {
		status := "❌ UNAVAILABLE"
		if available {
			status = "✅ AVAILABLE"
		}

		caps := d.capabilities[method]
		builder.WriteString(fmt.Sprintf("%-20s %s - %s\n",
			method, status, caps.Name))

		if caps.HardwareInfo != nil {
			builder.WriteString(fmt.Sprintf("                    Device: %s (%s)\n",
				caps.HardwareInfo.DevicePath,
				caps.HardwareInfo.ConnectionType))
			if caps.HardwareInfo.ChipCount > 0 {
				builder.WriteString(fmt.Sprintf("                    Chips: %d\n", caps.HardwareInfo.ChipCount))
			}
		}

		if !available && caps.Reason != "" {
			builder.WriteString(fmt.Sprintf("                    Reason: %s\n", caps.Reason))
		}
		builder.WriteString("\n")
	}

	availableCount := 0
	for _, available := range d.detectedMethods {
		if available {
			availableCount++
		}
	}

	builder.WriteString(fmt.Sprintf("Total Methods: %d\n", len(d.detectedMethods)))
	builder.WriteString(fmt.Sprintf("Available: %d\n", availableCount))

	return builder.String()
}
