param([switch]$PreflightOnly)
$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$linuxRoot='/mnt/f/FlashInferResearch/S03-M06'
$launchLock=$null
$runDir=$null
$wslProcess=$null
$launchId=$null
$heartbeatSeq=0
$lastPower=$null
$exitCode=1
function Write-AtomicJson($path,$value) {
    $temporary=$path+'.'+[Guid]::NewGuid().ToString('N')+'.tmp'
    [IO.File]::WriteAllText($temporary,($value|ConvertTo-Json -Depth 12 -Compress),[Text.UTF8Encoding]::new($false))
    try {
        for ($attempt=0;$attempt -lt 5;$attempt++) {
            try {
                if ([IO.File]::Exists($path)) { [IO.File]::Replace($temporary,$path,[NullString]::Value) }
                else { [IO.File]::Move($temporary,$path) }
                return
            } catch {
                if ($attempt -eq 4) { throw }
                Start-Sleep -Milliseconds 40
            }
        }
    } finally {
        if ([IO.File]::Exists($temporary)) { [IO.File]::Delete($temporary) }
    }
}

function Get-PowerSnapshot {
    $ac=[Windows.Forms.SystemInformation]::PowerStatus.PowerLineStatus.ToString()
    $text=(& powercfg.exe /getactivescheme | Out-String)
    if ($LASTEXITCODE -ne 0 -or $text -notmatch '(?i)[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}') {
        throw '读取Windows电源方案失败。'
    }
    return @{ac_online=($ac -eq 'Online');power_scheme=$matches[0].ToLowerInvariant();epoch=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000.0}
}

function Write-Heartbeat($power,[bool]$stop) {
    $script:heartbeatSeq++
    $record=@{launch_id=$script:launchId;seq=$script:heartbeatSeq;ac_online=[bool]$power.ac_online;power_scheme=[string]$power.power_scheme;stop=$stop;epoch=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000.0}
    Write-AtomicJson (Join-Path $script:runDir 'heartbeat.json') $record
}

function Read-LogTail($path) {
    if (-not [IO.File]::Exists($path)) { return }
    $stream=[IO.File]::Open($path,[IO.FileMode]::Open,[IO.FileAccess]::Read,([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete))
    try {
        $start=[math]::Max(0,$stream.Length-16384)
        $null=$stream.Seek($start,[IO.SeekOrigin]::Begin)
        $buffer=New-Object byte[] 16384
        $count=$stream.Read($buffer,0,$buffer.Length)
        $text=[Text.Encoding]::UTF8.GetString($buffer,0,$count)
        if ($start -gt 0) {
            $newline=$text.IndexOf("`n")
            if ($newline -ge 0) { $text=$text.Substring($newline+1) }
        }
        $text -split '\r?\n' | Select-Object -Last 12
    } finally { $stream.Dispose() }
}

try {
    try { $launchLock=[IO.File]::Open((Join-Path $root 'windows-launch.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None) }
    catch { throw 'S03-M06启动窗口已打开，请查看原窗口或查询进度。' }
    Add-Type -AssemblyName System.Windows.Forms
    $lastPower=Get-PowerSnapshot
    if (-not $PreflightOnly -and -not $lastPower.ac_online) { throw '请接通电源后启动长请求验证。' }
    $protocol=Get-Content -LiteralPath (Join-Path $root 'protocol.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $launchId=[Guid]::NewGuid().ToString()
    $runDir=Join-Path (Join-Path $root 'runs') $launchId
    $null=New-Item -ItemType Directory -Path $runDir
    Write-AtomicJson (Join-Path $root 'active-launch.json') @{launch_id=$launchId;run_dir=$runDir;preflight_only=[bool]$PreflightOnly}
    Write-Heartbeat $lastPower $false
    Write-Host 'FlashInfer S03-M06：1到2：受益场景下的正确性验收。' -ForegroundColor Cyan
    Write-Host ('4个新进程：默认→自动选择→自动选择→默认；四条16384输入；原128-token性能批与混合标准答案批分别记录。GPU累计最多'+[math]::Round([double]$protocol.gpu_budget_seconds/60,2)+'分钟，单进程最多5.5分钟，包含加载、Graph准备和测量；新增空间最多1GiB。')
    Write-Host '同一65%显存预算，缓存由vLLM自动分配；纯性能2次预热、5次测量；4个标准题场景各2次；每进程另做原6道题。'
    Write-Host ('本次启动记录：'+$runDir)
    if ($PreflightOnly) { Write-Host '本次只检查启动接口，不运行计算进程。' -ForegroundColor Cyan }
    $arguments=@('-d',[string]$protocol.distro,'--',[string]$protocol.python_linux,'-B',($linuxRoot+'/runner.py'),'--launch-id',$launchId,'--run-dir',($linuxRoot+'/runs/'+$launchId))
    if ($PreflightOnly) { $arguments += '--preflight-only' }
    $wslProcess=Start-Process -FilePath 'wsl.exe' -ArgumentList $arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runDir 'windows.log') -RedirectStandardError (Join-Path $runDir 'error.log')
    $null=$wslProcess.Handle
    while (-not $wslProcess.HasExited) {
        $lastPower=Get-PowerSnapshot
        Write-Heartbeat $lastPower $false
        Start-Sleep -Seconds 2
        $wslProcess.Refresh()
    }
    $wslProcess.WaitForExit()
    $exitCode=$wslProcess.ExitCode
    Write-Host ('本次启动已结束，退出码：'+$exitCode)
    if ($exitCode -eq 0) {
        $savedStatusPath=Join-Path $root 'status.json'
        $savedStatus=$null
        if ([IO.File]::Exists($savedStatusPath)) { $savedStatus=Get-Content -LiteralPath $savedStatusPath -Raw -Encoding UTF8 | ConvertFrom-Json }
        if ($savedStatus -and $savedStatus.state -eq 'gate_stopped') {
            Write-Host '显存闸门未通过，已保留前半轮结果，容量阶段未运行。' -ForegroundColor Yellow
        } elseif ($savedStatus -and $savedStatus.state -eq 'budget_stopped') {
            Write-Host '剩余额度已用完，已保存结果，后续步骤未自动追加。' -ForegroundColor Yellow
        } else { Write-Host '已保存结果，请点击“查询进度.cmd”。' -ForegroundColor Green }
    }
    elseif ($exitCode -eq 2) { Write-Host '已有进程运行或收尾，请查看原窗口。' -ForegroundColor Yellow }
    else {
        foreach($name in @('error.json','error.log','windows.log')) { Read-LogTail (Join-Path $runDir $name) }
        Write-Host '已完成的格与历史记录均保留，查看查询进度后反馈。' -ForegroundColor Yellow
    }
} catch {
    $exitCode=1
    Write-Host $_.Exception.Message -ForegroundColor Red
    if ($runDir) { $_|Out-String|Set-Content -LiteralPath (Join-Path $runDir 'launcher-error.log') -Encoding UTF8 }
} finally {
    if ($wslProcess) {
        try {
            $wslProcess.Refresh()
            if (-not $wslProcess.HasExited) {
                if ($lastPower) { Write-Heartbeat $lastPower $true }
                $null=$wslProcess.WaitForExit(25000)
            }
        } catch { }
        $wslProcess.Dispose()
    }
    if ($launchLock) { $launchLock.Dispose() }
}
exit $exitCode