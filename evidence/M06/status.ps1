param([string]$Root=$PSScriptRoot)
$ErrorActionPreference='Stop'
function Read-Record($name) {
    $path=Join-Path $Root $name
    if ([IO.File]::Exists($path)) { return (Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json) }
    return $null
}
Write-Host 'FlashInfer S03-M06：1到2，16K四并发正确性验收。' -ForegroundColor Cyan
Write-Host '同一Qwen、65%显存预算；四条16K请求；标准答案正常结束，三条陪跑各输出128。'
$s=Read-Record 'status.json'
$labels=@{not_started='等待点击启动';running='运行中';ready='本进程已保存';complete='四个进程的正确性与性能数据已采集完，可分析';stopped='已停止，保留记录待分析';budget_stopped='本轮额度已用完，结果已保留'}
if ($s) {
    $label=$labels[[string]$s.state]; if(-not $label){$label=[string]$s.state}
    Write-Host ('记录状态：'+$label)
    Write-Host ('已保存进程：'+$s.completed_jobs+'/4；已完成测试格：'+$s.completed_cases+'/4；默认→自动→自动→默认')
    if($s.current_job){Write-Host ('当前/最近进程：'+$s.current_job)}
    if($s.progress){
        $phases=@{loading_model_and_capturing_graphs='加载模型、准备执行图并查询工作区';quality_probes='原六道质量题';warming_up='长请求预热';measuring='长请求正式计时';case_complete='本格已保存';complete='本进程已保存';failed='本进程停止';resumed_completed_case='复用已完成测试格';concurrent_quality='四并发标准答案检查';concurrent_quality_case_complete='标准题场景已保存';resumed_concurrent_quality='复用已完成标准题场景'}
        $phase=$phases[[string]$s.progress.phase]; if(-not $phase){$phase=[string]$s.progress.phase}
        Write-Host ('当前阶段：'+$phase)
        if($s.progress.concurrent_case_id){Write-Host ('标准题场景：'+$s.progress.concurrent_case_id+'；已保存重复 '+[int]$s.progress.concurrent_repeat_done+'/2')}
        if($s.progress.case_id){
            $length=if([string]$s.progress.case_id -eq 'L12288_B4'){12288}elseif([string]$s.progress.case_id -eq 'L16384_B4'){16384}else{[string]$s.progress.case_id}
            Write-Host ('当前输入长度：'+$length+' token／请求；4条请求')
        }
        if($s.progress.phase -eq 'warming_up'){Write-Host ('预热已完成：'+[int]$s.progress.warmup_done+'/2')}
        if($s.progress.phase -eq 'measuring'){Write-Host ('正式测量已完成：'+[int]$s.progress.measurement_done+'/5')}
    }
    Write-Host ('GPU阶段已用（含当前进程）：'+[math]::Round([double]$s.gpu_observed_seconds,1)+'秒 / '+$s.gpu_budget_seconds+'秒')
    Write-Host ('本轮剩余：'+[math]::Round([double]$s.gpu_remaining_seconds,1)+'秒；当前进程已运行：'+[math]::Round([double]$s.live_elapsed_seconds,1)+'秒')
    if($s.combined_gpu_observed_seconds){Write-Host ('S03整条研究累计：'+[math]::Round([double]$s.combined_gpu_observed_seconds,1)+'秒 / '+$s.combined_gpu_budget_seconds+'秒')}
    if($s.last_update_epoch){$age=[math]::Max(0,[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000.0-[double]$s.last_update_epoch);Write-Host ('最近记录距今：'+[math]::Round($age,1)+'秒')}
    if($s.message){Write-Host ([string]$s.message)}
    if($s.attempt_dir){
        $attempt=[string]$s.attempt_dir
        if($attempt.StartsWith('/mnt/f/')){$attempt='F:\'+$attempt.Substring(7).Replace('/','\')}
        Write-Host ('当前/最近日志：'+(Join-Path $attempt 'worker.log'))
        $decisionPath=Join-Path $attempt 'startup-decision.json'
        if([IO.File]::Exists($decisionPath)){
            try {
                $decisionReport=Get-Content -LiteralPath $decisionPath -Raw -Encoding UTF8 | ConvertFrom-Json
                foreach($decision in @($decisionReport.decisions)){
                    Write-Host ('启动选择：'+$decision.status+'；实际预留 '+[math]::Round([double]$decision.chosen_bytes/1048576,2)+' MiB；原预留 '+[math]::Round([double]$decision.configured_bytes/1048576,2)+' MiB')
                    if($decision.fallback){Write-Host ('沿用原配置的原因：'+[string]$decision.fallback)}
                }
            } catch { Write-Host '启动选择记录正在更新。' }
        }
        foreach($caseId in @('L16384_B4')){
            $casePath=Join-Path $attempt ('cases\'+$caseId+'.json')
            if([IO.File]::Exists($casePath)){
                try {
                    $c=Get-Content -LiteralPath $casePath -Raw -Encoding UTF8 | ConvertFrom-Json
                    Write-Host ($caseId+'：本次尝试已保存正式样本 '+[int]$c.measurement_count+'/5；完整格 '+[bool]$c.passed)
                } catch { Write-Host ($caseId+'：样本记录正在更新。') }
            }else{Write-Host ($caseId+'：本次尝试尚未保存正式样本')}
        }
        foreach($qaId in @('CQ_LOOKUP_SLOT4','CQ_LATEST_SLOT4','CQ_COMBINE_SLOT4','CQ_LOOKUP_SLOT1')){
            $qaPath=Join-Path $attempt ('concurrent-quality\'+$qaId+'.json')
            if([IO.File]::Exists($qaPath)){
                try {$qa=Get-Content -LiteralPath $qaPath -Raw -Encoding UTF8 | ConvertFrom-Json;Write-Host ($qaId+'：已保存 '+@($qa.samples).Count+'/2 次；采集完整 '+[bool]$qa.collection_complete)}catch{Write-Host ($qaId+'：记录正在更新。')}
            }
        }
        if($s.state -eq 'stopped' -and [IO.File]::Exists((Join-Path $attempt 'worker.log'))){Get-Content -LiteralPath (Join-Path $attempt 'worker.log') -Tail 8 -Encoding UTF8}
    }
}else{Write-Host '记录状态：等待点击启动。'}
$summary=Read-Record 'summary.json'
if($summary -and $summary.collection_complete){Write-Host '正确性与性能采集完成，请查看汇总文件并交回分析。'}
Write-Host ('结果文件：'+(Join-Path $Root 'summary.md'))
Write-Host '本入口只读取Windows文件。每次点击显示一次进度；旧记录时间仅供判断更新情况。'
exit 0
