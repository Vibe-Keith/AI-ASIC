# AI-ASIC launcher for Windows PowerShell. Examples:
#   .\run.ps1 detect
#   .\run.ps1 infer "hello world"
if (Get-Command py -ErrorAction SilentlyContinue) {
  py -m ai_asic.cli @args
} else {
  python -m ai_asic.cli @args
}
