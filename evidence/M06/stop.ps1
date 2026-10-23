param([string]$Root=$PSScriptRoot)
$ErrorActionPreference='Stop'
$launchId=$null
$activePath=Join-Path $Root 'active-launch.json'
if (Test-Path -LiteralPath $activePath) {
    try { $active=Get-Content -LiteralPath $activePath -Raw -Encoding UTF8 | ConvertFrom-Json; $launchId=$active.launch_id } catch { }
}
$path=Join-Path $Root 'stop.flag'
$temporary=$path+'.'+[Guid]::NewGuid().ToString('N')+'.tmp'
$record=@{launch_id=$launchId;requested_epoch=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000.0;reason='user_requested'}
[IO.File]::WriteAllText($temporary,($record|ConvertTo-Json -Compress),[Text.UTF8Encoding]::new($false))
try {
    for ($attempt=0;$attempt -lt 5;$attempt++) {
        try {
            if ([IO.File]::Exists($path)) { [IO.File]::Replace($temporary,$path,[NullString]::Value) }
            else { [IO.File]::Move($temporary,$path) }
            break
        } catch { if ($attempt -eq 4) { throw }; Start-Sleep -Milliseconds 40 }
    }
} finally { if ([IO.File]::Exists($temporary)) { [IO.File]::Delete($temporary) } }
Write-Host '停止请求已保存。运行窗口会停止当前任务并保存状态。' -ForegroundColor Cyan
Write-Host '请等运行窗口显示结束，再关闭窗口或继续运行。'
Write-Host '查询进度可以查看已保存步骤和各项累计预算。'
exit 0
