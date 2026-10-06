# AI-ASIC graphical interface launcher for Windows PowerShell.
#   .\run-gui.ps1
if (Get-Command py -ErrorAction SilentlyContinue) {
  py -m ai_asic.gui @args
} else {
  python -m ai_asic.gui @args
}
