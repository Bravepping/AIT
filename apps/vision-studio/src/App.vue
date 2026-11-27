<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, shallowRef, watch } from 'vue'
import { Channel, invoke, isTauri } from '@tauri-apps/api/core'
import { engines, type Engine, type Command, type MonitorCommand, type WorkerEvent, type Stream,
  type Inference, type Backend, type Gpu, type KmBox, type DryRun } from './protocol'
import { classIds, newProfile, migrateProfile, makeBackup, parseBackup, type Profile, type Preferences, type Backup } from './configuration'

const engineOptions = engines
const pages = ['总览', '设备连接', '推理设置', '热键管理', '目标识别', '鼠标控制', '运行日志'] as const
const page = ref<typeof pages[number]>('总览')
const pointerKeys = ['左键', '右键', '中键', 'Mouse4', 'Mouse5']
const previewRates = [5, 10, 15, 30, 60]
type Source = { name: string; url: string; key: string }
const profilesKey = 'vision-v11-profiles'
function saved<T>(key: string, fallback: T): T {
  try { return JSON.parse(localStorage.getItem(key) ?? 'null') ?? fallback }
  catch { return fallback }
}
const profiles = ref(saved(profilesKey, [newProfile(1)]))
if (!Array.isArray(profiles.value) || !profiles.value.length) profiles.value = [newProfile(1)]
profiles.value = profiles.value.map(migrateProfile)
const selection = ref(saved('vision-v17-selection', profiles.value[0]!.id))
const profile = computed(() => profiles.value.find((entry) => entry.id === selection.value) ?? profiles.value[0]!)
const enabledClasses = ref(saved('vision-v11-enabled-classes', [...classIds]))
const preferencesKey = 'vision-v13-settings'
const preferences = saved<Preferences>(preferencesKey, {
  boxAddress: '', boxPort: '' as number | '', boxUuid: '', monitorPort: 12345,
  engine: 'TensorRT' as Engine, model: '1013v26s_640.onnx', source: '', preview: 30,
})
const boxAddress = ref(preferences.boxAddress ?? '')
const boxPort = ref<number | ''>(preferences.boxPort ?? '')
const boxUuid = ref(preferences.boxUuid ?? '')
const monitorPort = ref(preferences.monitorPort ?? 12345)
const boxError = ref('')
const configBusy = ref(false)
const backupInput = ref<HTMLInputElement | null>(null)
const previousBackup = ref(saved<Backup | null>('vision-v17-previous-backup', null))
const configStatus = ref('')
const configFailed = ref(false)
const inputPairing = ref('')
const kmBox = shallowRef<KmBox | null>(null)
const dryRun = shallowRef<DryRun>({ kind: 'dry_run', state: 'Idle', armed: false, active: false,
  reason: '等待实时状态', candidate: null, mode: 'dry_run', real_control_available: false })
const dryError = ref('')
const observedX = ref<number | ''>('')
const observedY = ref<number | ''>('')
const desktopWidth = ref<number | ''>('')
const desktopHeight = ref<number | ''>('')
const controlMode = ref<'dry_run' | 'calibration' | 'real'>('dry_run')
const dryRunPassed = ref(false)
const stepPassed = ref(false)
const fullPassed = ref(false)
const realArming = ref(false)
const calibrationStatus = ref('')
const calibrationFailed = ref(false)
const profileError = ref('')
const stateNames: Record<DryRun['state'], string> = {
  Idle: '等待条件', Ready: '已就绪', Armed: '已启用', Running: '跟随计算中', Stopped: '已停止', Fault: '故障锁止',
}
const conflicts = computed(() => profiles.value.filter((entry) => entry.enabled && profiles.value.some((other) =>
  other.enabled && other.id !== entry.id && other.mask === entry.mask)))
const candidate = computed(() => dryRun.value.candidate)
const backend = shallowRef<Backend | null>(null)
const gpu = shallowRef<Gpu | null>(null)
const workerState = ref('starting')
const workerError = ref('')
const providers = ref<Engine[]>([])
const requestedEngine = ref<Engine>(engineOptions.includes(preferences.engine) ? preferences.engine : 'TensorRT')
const activeEngine = ref('')
const modelStatus = ref('模型未加载')
const loadingModel = ref(false)
const selectedModel = ref(preferences.model || '1013v26s_640.onnx')
const modelFiles = ref<string[]>([])
const discovering = ref(false)
const sourceOptions = ref<Source[]>([])
const sourceSelection = ref('')
const preferredSource = ref(preferences.source ?? '')
const streamState = ref('idle')
const streamLabel = ref('未连接 NDI')
const stream = shallowRef<Stream | null>(null)
const rangeDiagram = computed(() => {
  const item = candidate.value
  const [width, height] = item?.dimensions ?? [stream.value?.width ?? 0, stream.value?.height ?? 0]
  if (!width || !height) return null
  const [x, y] = item?.reference ?? [width * profile.value.referenceX / 100, height * profile.value.referenceY / 100]
  const radius = item?.range_px ?? Math.min(width, height) * (profile.value.dynamicRange ?? profile.value.rangePercent) / 200
  return {
    aspectRatio: `${width} / ${height}`,
    ring: { left: `${x / width * 100}%`, top: `${y / height * 100}%`,
      width: `${2 * radius / width * 100}%`, height: `${2 * radius / height * 100}%` },
    anchor: item ? { left: `${item.normalized[0] * 100}%`, top: `${item.normalized[1] * 100}%` } : null,
  }
})
const inference = shallowRef<(Inference & { fps?: number }) | null>(null)
const previewRate = ref(0)
const preferredPreview = ref(previewRates.includes(preferences.preview) ? preferences.preview : 30)
const previewImage = ref('')
const previewFrames = ref(0)
const cudaChecking = ref(false)
const cudaStatus = ref('')
const logs = shallowRef<{ at: Date; text: string; level: string }[]>([])
const logPath = ref('')
const logError = ref('')
const deviceAOperation = ref(false)

let session = 0
let sourceRevision = 0
let modelRevision = 0
let previewRevision = 0
let recentInference: { at: number; runs: number } | null = null
let channel: Channel<WorkerEvent> | null = null
let nextHealthLog = 0
let nextCandidateLog = 0
let controlsRevision = 0
let profileSync: Promise<boolean> = Promise.resolve(true)

const ready = computed(() => workerState.value === 'ready')
const selectedSource = computed(() =>
  sourceOptions.value.find((source) => source.key === sourceSelection.value),
)
const resolution = computed(() =>
  stream.value?.width && stream.value?.height
    ? `${stream.value.width} × ${stream.value.height}`
    : '—',
)
const modelAverage = computed(() => {
  const stats = inference.value
  return stats?.warm_samples !== undefined ? stats.model_avg : stats?.average
})

function format(value: number | null | undefined, precision = 1) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(precision) : '—'
}

function addLog(text: string, level = 'info', at = new Date()) {
  if (!text) return
  logs.value = [...logs.value.slice(-1499), { at, text, level }]
}

function markDeviceAOperation() {
  deviceAOperation.value = !deviceAOperation.value
  addLog(`设备 A 人工操作 · ${deviceAOperation.value ? '开始' : '结束'}（人工标记）`)
}

function asText(error: unknown) {
  return error instanceof Error ? error.message : String(error)
}

function resetStreams() {
  sourceRevision++
  streamState.value = 'idle'
  streamLabel.value = '未连接 NDI'
  stream.value = null
  previewImage.value = ''
  inference.value = null
  recentInference = null
}

async function launch() {
  const current = ++session
  resetStreams()
  providers.value = []
  sourceOptions.value = []
  sourceSelection.value = ''
  activeEngine.value = ''
  backend.value = null
  modelStatus.value = '模型未加载'
  loadingModel.value = false
  discovering.value = false
  cudaChecking.value = false
  cudaStatus.value = ''
  workerError.value = ''
  workerState.value = 'starting'
  previewRate.value = 0
  previewFrames.value = 0
  if (!isTauri()) {
    workerState.value = 'failed'
    workerError.value = '请在 NDI Vision Control 桌面应用中打开此界面。'
    return
  }
  channel = new Channel<WorkerEvent>()
  channel.onmessage = (message) => {
    if (current === session) readEvent(message)
  }
  try {
    await invoke('start_worker', { events: channel })
  } catch (error) {
    workerState.value = 'failed'
    workerError.value = asText(error)
    addLog(workerError.value, 'error')
  }
}

async function send(command: Command) {
  try {
    await invoke('send_worker', { command })
    return true
  } catch (error) {
    addLog(asText(error), 'error')
    return false
  }
}

async function discover() {
  discovering.value = true
  if (!(await send({ cmd: 'discover' }))) discovering.value = false
}

async function changeStream(source: Source | null | undefined) {
  resetStreams()
  if (source) {
    streamState.value = 'connecting'
    streamLabel.value = `等待画面 · ${source.name}`
  }
  const ok = await send({
    cmd: 'stream',
    source: sourceRevision,
    name: source?.name ?? '',
    url: source?.url ?? '',
  })
  if (!ok) {
    streamState.value = 'error'
    streamLabel.value = 'NDI 操作失败，详情见运行日志'
  }
  if (ok && source) {
    preferredSource.value = source.name
    if (!activeEngine.value && !loadingModel.value) await loadModel()
  }
}

async function loadModel() {
  modelRevision++
  const provider = requestedEngine.value
  activeEngine.value = ''
  backend.value = null
  loadingModel.value = true
  modelStatus.value = `正在加载 ${provider}`
  inference.value = null
  recentInference = null
  previewImage.value = ''
  const ok = await send({ cmd: 'model', model: modelRevision, provider, filename: selectedModel.value })
  if (!ok) {
    loadingModel.value = false
    modelStatus.value = `${provider} 加载请求失败，详情见运行日志`
  }
}

async function refreshModels() {
  try {
    const files = await invoke<string[]>('list_models')
    modelFiles.value = files
    if (files.length && !files.includes(selectedModel.value)) selectedModel.value = files[0]!
  } catch (error) {
    addLog(asText(error), 'error')
  }
}

async function setPreview(fps: number) {
  const previousRate = previewRate.value
  previewImage.value = ''
  previewRate.value = fps
  const revision = ++previewRevision
  const ok = await send({ cmd: 'preview', fps, preview: revision })
  if (!ok && revision === previewRevision) previewRate.value = previousRate
}

async function checkCuda() {
  cudaChecking.value = true
  cudaStatus.value = '正在检查 CUDA 环境'
  if (!(await send({ cmd: 'cuda_check' }))) {
    cudaChecking.value = false
    cudaStatus.value = 'CUDA 自检启动失败，详情见运行日志'
  }
}

async function exportLog() {
  logError.value = ''
  const text = logs.value
    .map((entry) => `${entry.at.toISOString()}  ${entry.text}`)
    .join('\n')
  try {
    logPath.value = await invoke<string>('export_log', { text })
  } catch (error) {
    logError.value = asText(error)
  }
}

function updateInference(message: Inference) {
  const now = performance.now()
  let fps = inference.value?.fps
  if (recentInference == null) {
    recentInference = { at: now, runs: message.runs }
  } else if (now - recentInference.at >= 1000) {
    fps = ((message.runs - recentInference.runs) * 1000) / (now - recentInference.at)
    recentInference = { at: now, runs: message.runs }
  }
  inference.value = { ...message, fps }
  if (now >= nextHealthLog && message.warm_samples && message.receiver_avg !== undefined) {
    nextHealthLog = now + 10000
    addLog(`${activeEngine.value} · 输入 ${format(stream.value?.fps)} FPS · 推理 ${format(message.inference_fps)} FPS · 接收到坐标 ${format(message.receiver_avg, 2)} / P95 ${format(message.receiver_p95)} / P99 ${format(message.receiver_p99)} ms · 显存 ${gpu.value?.used_mib ?? '—'} MiB · 预览 ${previewRate.value} FPS`)
  }
}

async function monitor(command: MonitorCommand) {
  boxError.value = ''
  try { await invoke('send_monitor', { command }) }
  catch (error) { boxError.value = asText(error) }
}

async function openKmBoxAcceptance() {
  boxError.value = ''
  try {
    await invoke('open_kmbox_acceptance')
    kmBox.value = null
    addLog('已打开实体鼠标验收，主程序只读监控已关闭')
  } catch (error) {
    boxError.value = asText(error)
    addLog(`实体鼠标验收 · ${boxError.value}`, 'error')
  }
}

async function applyCalibration() {
  calibrationStatus.value = ''
  calibrationFailed.value = false
  const id = profile.value.id
  const source = preferredSource.value
  const device = boxUuid.value.replace(/[:\-]/g, '').trim().toUpperCase()
  const crop = stream.value ? [stream.value.width, stream.value.height] as [number, number] : undefined
  const desktop = [Number(desktopWidth.value), Number(desktopHeight.value)] as [number, number]
  const siteReady = streamState.value === 'live' && source && crop?.every((side, index) =>
    side > 0 && Number.isInteger(desktop[index]) && desktop[index]! >= side) && /^[0-9A-F]{8}$/.test(device)
  const measured = { observedX: Number(observedX.value), observedY: Number(observedY.value),
    ...(siteReady ? { source, device, desktop, crop } : {}) }
  try {
    const result = await invoke<{ gain_x: number; gain_y: number; invert_x: boolean; invert_y: boolean }>(
      'estimate_calibration', measured)
    const current = profiles.value.find((entry) => entry.id === id)
    if (!current) throw new Error('该校准方案已删除')
    const updated = { ...current, gainX: result.gain_x, gainY: result.gain_y,
      invertX: result.invert_x, invertY: result.invert_y, calibration: measured }
    const next = profiles.value.map((entry) => entry.id === id ? updated : entry)
    localStorage.setItem(profilesKey, JSON.stringify(next))
    profiles.value = next
    calibrationStatus.value = `已保存 ${current.name} 的坐标换算 · 实测 X ${measured.observedX} / Y ${measured.observedY} px · 比例 X ${format(result.gain_x, 6)} / Y ${format(result.gain_y, 6)}`
    addLog(calibrationStatus.value)
  } catch (error) {
    calibrationFailed.value = true
    calibrationStatus.value = asText(error)
  }
}

function profilePayload(entries: Profile[], classes: number[]) {
  return entries.filter((entry) => entry.enabled).map((entry) => ({
    id: entry.id, name: entry.name, mask: entry.mask, toggle: entry.mode === 'toggle',
    event_preview: entry.mode === 'hold' && entry.eventPreview === true,
    enabled_classes: [...classes],
    classes: entry.scope === 'all' ? null : entry.scope === 'single' ? [entry.single] : [...entry.classes],
    priority: entry.priority, confidence: entry.confidence / 100, anchor_x: entry.anchorX, anchor_y: entry.anchorY,
    range_percent: entry.rangePercent,
    gain_x: entry.gainX, gain_y: entry.gainY, invert_x: entry.invertX, invert_y: entry.invertY, max_step: entry.maxStep,
    dead_zone_px: entry.deadZonePx, smooth_half_life_ms: entry.smoothHalfLifeMs,
    switch_confirm_frames: entry.switchConfirmFrames,
    lost_wait_ms: entry.lostWaitMs, dynamic_range_percent: entry.dynamicRange,
    per_class: Object.fromEntries(Object.entries(entry.classRules).map(([id, rule]) => [id, {
      threshold: (rule.threshold === '' ? entry.confidence : rule.threshold) / 100,
      priority: rule.priority, anchor: rule.anchor,
    }])),
    dead_zone_axes: entry.deadZoneAxes, max_speed: entry.maxSpeed, acceleration: entry.acceleration,
    tracking_gain: entry.trackingGain,
    motion: { algorithm: entry.motion.algorithm, derivative_ms: entry.motion.derivativeMs,
      prediction_ms: entry.motion.predictionMs, slow_radius_px: entry.motion.slowRadiusPx, adaptive: entry.motion.adaptive },
    reference_x: entry.referenceX, reference_y: entry.referenceY,
    calibration_site: entry.calibration?.desktop && entry.calibration.crop
      ? { source: entry.calibration.source, device: entry.calibration.device,
          desktop: entry.calibration.desktop, crop: entry.calibration.crop,
          observed: [entry.calibration.observedX, entry.calibration.observedY] } : undefined,
  }))
}

function syncProfiles() {
  const configs = profilePayload(profiles.value, enabledClasses.value)
  profileSync = profileSync.then(async () => {
    try {
      await invoke('configure_gate', { profiles: configs })
      profileError.value = ''
      return true
    } catch (error) {
      profileError.value = asText(error)
      return false
    }
  })
  return profileSync
}

function snapshotConfiguration(): Backup {
  return makeBackup(profiles.value, enabledClasses.value, selection.value, {
    boxAddress: boxAddress.value, boxPort: boxPort.value, boxUuid: boxUuid.value, monitorPort: monitorPort.value,
    engine: requestedEngine.value, model: selectedModel.value, source: preferredSource.value, preview: preferredPreview.value,
  })
}

function exportConfiguration() {
  configFailed.value = false
  const blob = new Blob([JSON.stringify(snapshotConfiguration(), null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = 'ndi-vision-v17-configuration.json'
  link.click()
  window.setTimeout(() => URL.revokeObjectURL(url), 1500)
  configStatus.value = '已请求保存配置文件'
}

async function applyConfiguration(serialized: string) {
  const backup = parseBackup(serialized)
  await invoke('validate_profile_backup', { profiles: profilePayload(backup.profiles, backup.enabledClasses) })
  await profileSync
  controlsRevision++
  await invoke('stop_gate')
  const existing = snapshotConfiguration()
  const previous = JSON.stringify(existing)
  const { preferences: next } = backup
  if (kmBox.value?.connected && (
    next.boxAddress !== existing.preferences.boxAddress || next.boxPort !== existing.preferences.boxPort ||
    next.boxUuid !== existing.preferences.boxUuid || next.monitorPort !== existing.preferences.monitorPort)) {
    await invoke('send_monitor', { command: { cmd: 'disconnect' } })
  }
  if (streamState.value !== 'idle' && next.source !== existing.preferences.source) await changeStream(null)
  localStorage.setItem('vision-v17-previous-backup', previous)
  previousBackup.value = JSON.parse(previous)
  dryRunPassed.value = stepPassed.value = fullPassed.value = false
  controlMode.value = 'dry_run'
  profiles.value = backup.profiles
  enabledClasses.value = backup.enabledClasses
  selection.value = backup.selection
  boxAddress.value = next.boxAddress
  boxPort.value = next.boxPort
  boxUuid.value = next.boxUuid
  monitorPort.value = next.monitorPort
  requestedEngine.value = next.engine
  selectedModel.value = next.model
  preferredSource.value = next.source
  preferredPreview.value = next.preview
  await nextTick()
  if (ready.value && !(await syncProfiles())) throw new Error(profileError.value)
}

async function changeConfiguration(read: () => Promise<string>, success: string) {
  if (configBusy.value) return
  configBusy.value = true
  configFailed.value = false
  try {
    await applyConfiguration(await read())
    configStatus.value = success
    addLog(success)
  } catch (error) {
    configFailed.value = true
    configStatus.value = asText(error)
  } finally { configBusy.value = false }
}

function importConfiguration(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = ''
  if (!file) return
  return changeConfiguration(async () => {
    if (file.size > 1024 * 1024) throw new Error('配置文件超出 1 MB')
    return file.text()
  }, '已导入配置，真实跟随需要重新武装')
}

async function setDryRun(enabled: boolean, emergency = false) {
  if (enabled && configBusy.value) return
  const revision = ++controlsRevision
  dryError.value = ''
  try {
    if (!enabled) await invoke('stop_gate', { emergency })
    else if (await profileSync && revision === controlsRevision) await invoke('arm_gate')
  } catch (error) {
    dryError.value = asText(error)
    addLog(`Dry Run · ${dryError.value}`, 'error')
  }
}

async function armReal() {
  if (realArming.value || configBusy.value) return
  realArming.value = true
  const revision = ++controlsRevision
  dryError.value = ''
  try {
    if (await profileSync && revision === controlsRevision) {
      await invoke('arm_real_tracking', { profileId: profile.value.id, dryRunPassed: dryRunPassed.value,
        stepPassed: stepPassed.value, fullPassed: fullPassed.value })
    }
  } catch (error) {
    dryError.value = asText(error)
    addLog(`真实跟随 · ${dryError.value}`, 'error')
  } finally {
    realArming.value = false
  }
}

function pressKey(index: number, checked: boolean) {
  profile.value.mask = checked ? profile.value.mask | (1 << index) : profile.value.mask & ~(1 << index)
  if ((profile.value.mask & 3) === 3) profile.value.mode = 'hold'
}

function addProfile() {
  const id = Math.max(...profiles.value.map((entry) => entry.id)) + 1
  profiles.value.push({ ...newProfile(id), enabled: false, mask: 0 })
  selection.value = id
}

function deleteProfile() {
  if (profiles.value.length < 2) return
  profiles.value = profiles.value.filter((entry) => entry.id !== selection.value)
  selection.value = profiles.value[0]!.id
}

function readEvent(message: WorkerEvent) {
  switch (message.kind) {
    case 'ready': {
      const reported = message.providers ?? ['CUDA', 'CPU']
      providers.value = engineOptions.filter((engine) => reported.includes(engine))
      if (!providers.value.includes(requestedEngine.value)) addLog(`当前环境未报告 ${requestedEngine.value}，请检查运行库或切换引擎`, 'error')
      workerState.value = 'ready'
      previewFrames.value = message.preview_frames ?? 0
      addLog('推理进程已启动，预览关闭')
      void syncProfiles()
      void refreshModels()
      void invoke('report_frontend_ready')
      break
    }
    case 'worker_exit':
      workerState.value = 'failed'
      workerError.value = '推理进程已退出。请查看日志后重启进程。'
      loadingModel.value = discovering.value = cudaChecking.value = false
      activeEngine.value = ''
      modelStatus.value = '模型未加载'
      resetStreams()
      addLog(workerError.value, 'error')
      break
    case 'sources':
      discovering.value = false
      if (message.error) {
        addLog(message.error, 'error')
        if (streamState.value === 'idle') streamLabel.value = 'NDI 发现失败，详情见运行日志'
        break
      }
      sourceOptions.value = message.sources.map(([name, url], index) => ({
        name, url, key: String(index),
      }))
      sourceSelection.value = (sourceOptions.value.find((source) => source.name === preferredSource.value) ??
        sourceOptions.value[0])?.key ?? ''
      addLog(`发现 ${sourceOptions.value.length} 个 NDI 源`)
      if (!sourceOptions.value.length && streamState.value === 'idle') streamLabel.value = '未发现 NDI 源'
      break
    case 'stream_state':
      if (message.source !== sourceRevision) break
      if (message.state === 'lost') {
        streamState.value = 'lost'
        streamLabel.value = `NDI 断流 · ${message.gap_ms} ms 内未收到新帧`
        inference.value = null
        recentInference = null
        previewImage.value = ''
      } else {
        streamState.value = 'live'
        streamLabel.value = message.state === 'resumed' ? 'NDI 已恢复' : 'NDI 接收中'
      }
      addLog(streamLabel.value)
      break
    case 'stream':
      if (message.source !== sourceRevision) break
      if (message.error) {
        streamState.value = 'error'
        streamLabel.value = message.error
        inference.value = null
        recentInference = null
        previewImage.value = ''
        addLog(message.error, 'error')
        break
      }
      stream.value = message
      previewFrames.value = message.preview_frames ?? previewFrames.value
      break
    case 'model_loading':
      if (message.model === modelRevision) modelStatus.value = message.status ?? '正在初始化推理引擎'
      break
    case 'model':
      if (message.model !== modelRevision) break
      loadingModel.value = false
      activeEngine.value = message.error ? '' : message.provider ?? ''
      backend.value = message.backend ?? null
      modelStatus.value = message.error
        ? `${message.requested ?? requestedEngine.value} · ${message.error}`
        : `${activeEngine.value} · ${message.metadata}`
      const prefix = message.error ? (message.phase === 'inference' ? '推理计算异常 · ' : '模型加载异常 · ') : ''
      addLog(prefix + modelStatus.value, message.error ? 'error' : 'info')
      if (message.backend) {
        addLog(`${activeEngine.value} ${message.backend.version ?? ''} · ${message.backend.precision ?? ''}` +
          (message.backend.cache ? ` · 引擎缓存 ${message.backend.cache}` : ''))
        for (const attempt of message.backend.previous_attempts ?? []) addLog(`${attempt.provider} 失败 · ${attempt.error}`, 'error')
      }
      if (!activeEngine.value) {
        inference.value = null
        recentInference = null
      }
      break
    case 'inference':
      if (message.source === sourceRevision && message.model === modelRevision && streamState.value === 'live') {
        updateInference(message)
      }
      break
    case 'preview_state':
      if (message.preview !== previewRevision) break
      previewRate.value = message.fps
      previewFrames.value = message.frames
      if (message.error) addLog(message.error, 'error')
      addLog(message.fps ? `预览开启 · 上限 ${message.fps} FPS` : '预览关闭')
      break
    case 'preview':
      if (previewRate.value && message.source === sourceRevision &&
          message.preview === previewRevision && streamState.value === 'live') {
        previewImage.value = `data:image/jpeg;base64,${message.image}`
      }
      break
    case 'cuda_check':
      cudaChecking.value = false
      cudaStatus.value = message.status
      addLog(message.status)
      for (const line of message.details ?? []) addLog(line)
      break
    case 'gpu':
      gpu.value = message
      break
    case 'kmbox':
      kmBox.value = message
      break
    case 'rate_dip': {
      const phase = message.duration_ms === null ? '下降' : `恢复 · 持续约 ${format(message.duration_ms / 1000, 2)} s`
      const target = message.target_sequence === null ? '无有效目标' :
        `目标帧 ${message.target_sequence} · 更新于 ${new Date(message.target_updated_at_ms ?? 0).toLocaleTimeString('zh-CN', { hour12: false })} · ${message.target_age_ms} ms 前`
      const category = message.metric === 'input' ? 'NDI 输入波动' : '推理吞吐波动'
      const operation = deviceAOperation.value ? ' · 设备 A 人工操作时段' : ''
      addLog(`${category} · FPS ${phase} · ${format(message.fps)} FPS（阈值 ${format(message.threshold)}） · 视频帧龄 ${message.frame_age_ms ?? '—'} ms · 待处理 原帧 ${message.raw_pending ?? '—'} / 已预处理 ${message.prepared_pending ?? '—'} · ${target}${operation}`)
      break
    }
    case 'kmbox_error':
      boxError.value = message.error
      addLog(message.error, 'error')
      break
    case 'kmbox_exit':
      kmBox.value = null
      boxError.value = 'KMBox 监控进程已退出'
      addLog(boxError.value, 'error')
      break
    case 'move_receipt':
      addLog(`KMBox #${message.id} · 锁定 ${message.target} · 类别 ${message.class} · dx/dy (${message.dx}, ${message.dy}) · ${message.status} · 队列 ${format(message.queue_ms, 2)} ms · 交换 ${message.exchange_ms} ms · 收帧到结果 ${message.receiver_ms} ms`,
        message.status === 'acknowledged' ? 'info' : 'error')
      break
    case 'dry_run': {
      const previous = dryRun.value
      dryRun.value = message
      const stop = message.last_stop
      if (stop && stop.id !== previous.last_stop?.id) {
        const mode = stop.mode === 'real' ? '真实跟随' : 'Dry Run'
        addLog(`${mode} · ${stateNames[message.state]} · ${stop.reason} · 停止前 ${stateNames[stop.previous_state]}` +
          ` · NDI 帧龄 ${stop.video_age_ms ?? '—'} ms · 推理帧龄 ${stop.inference_age_ms ?? '—'} ms（上限 ${stop.age_limit_ms} ms）` +
          ` · 帧 ${stop.frame_sequence ?? '—'} · Track ID ${stop.track_id ?? '—'} · 按键报文龄 ${stop.button_age_ms ?? '—'} ms` +
          ` · ${stop.cleared_pending ? '已清除待发送位移' : '队列为空'}`, message.state === 'Fault' ? 'error' : 'info', new Date(stop.at_ms))
      } else if (previous.state !== message.state || previous.profile_id !== message.profile_id) {
        addLog(`${message.mode === 'real' ? '真实跟随' : 'Dry Run'} · ${stateNames[message.state]} · ${message.reason}` +
          (message.profile_name ? ` · ${message.profile_name}` : ''), message.state === 'Fault' ? 'error' : 'info')
      }
      if (message.hold_confirmed && !previous.hold_confirmed) addLog('Dry Run · 设备 A 已确认当前热键处于按下状态')
      if (message.event_hold_age_ms != null && previous.event_hold_age_ms == null) addLog('Dry Run · 按上次按键事件保持候选，等待松键报文')
      if (message.candidate && (performance.now() >= nextCandidateLog || !previous.candidate ||
          previous.profile_id !== message.profile_id || previous.candidate.lock_id !== message.candidate.lock_id)) {
        const item = message.candidate
        addLog(`${message.mode === 'real' ? '真实跟随' : 'Dry Run'} · ${message.profile_name} (#${item.profile_id}) · 帧 ${item.sequence} · 类别 ${item.class} · 置信度 ${format(item.score * 100)}% · ${item.selection_reason} · 锁定点 (${item.anchor.map((v) => format(v)).join(', ')}) · 换算比例 X/Y ${item.signed_scale.map(v => format(v, 6)).join(' / ')} · dx/dy (${item.relative.join(', ')}) · 选取 ${format(item.selection_ms, 3)} / 位移 ${format(item.movement_ms, 3)} ms · 收帧到候选 ${item.latency_ms} ms`)
        nextCandidateLog = performance.now() + 1000
      }
      break
    }
    case 'log':
      addLog(message.text)
      break
  }
}

watch(() => [profile.value.id, profile.value.calibration] as const, ([, measurement]) => {
  observedX.value = measurement?.observedX ?? ''
  observedY.value = measurement?.observedY ?? ''
  desktopWidth.value = measurement?.desktop?.[0] ?? ''
  desktopHeight.value = measurement?.desktop?.[1] ?? ''
}, { immediate: true })

watch(controlMode, (mode) => {
  if (mode !== 'real' && (dryRun.value.mode === 'real' || realArming.value)) void setDryRun(false)
})

watch([profiles, enabledClasses], () => {
  controlsRevision++
  if (dryRun.value.armed) void invoke('stop_gate').catch((error) => { dryError.value = asText(error) })
  localStorage.setItem(profilesKey, JSON.stringify(profiles.value))
  localStorage.setItem('vision-v11-enabled-classes', JSON.stringify(enabledClasses.value))
  if (ready.value && !configBusy.value) void syncProfiles()
}, { deep: true })

watch(selection, (id) => localStorage.setItem('vision-v17-selection', JSON.stringify(id)))

watch([boxAddress, boxPort, boxUuid, monitorPort, requestedEngine, selectedModel, preferredSource, preferredPreview], () => {
  localStorage.setItem(preferencesKey, JSON.stringify({ boxAddress: boxAddress.value, boxPort: boxPort.value,
    boxUuid: boxUuid.value, monitorPort: monitorPort.value, engine: requestedEngine.value, model: selectedModel.value,
    source: preferredSource.value, preview: preferredPreview.value }))
})
watch(preferredPreview, (fps) => { if (previewRate.value) void setPreview(fps) })

function stopOnF8(event: KeyboardEvent) {
  if (event.key === 'F8') {
    event.preventDefault()
    if (!event.repeat) void setDryRun(false, true)
  }
}

onMounted(() => {
  window.addEventListener('keydown', stopOnF8)
  void launch()
})
onUnmounted(() => window.removeEventListener('keydown', stopOnF8))
</script>

<template>
  <header class="topbar">
    <div class="brand"><span class="brand-mark" aria-hidden="true"></span>NDI Vision Control <small>V1.7 Final 候选</small></div>
    <div class="top-actions">
      <div class="process-state"><span class="dot" :class="workerState"></span>
        {{ ready ? '设备 B · 推理进程就绪' : workerState === 'starting' ? '启动推理进程' : '推理进程不可用' }}
      </div>
      <button class="button danger" type="button" @click="setDryRun(false)">停止跟随 · F8</button>
    </div>
  </header>

  <div class="shell">
    <nav class="navigation" aria-label="主导航">
      <button v-for="item in pages" :key="item" type="button" :class="{ selected: page === item }"
        :aria-current="page === item ? 'page' : undefined" @click="page = item">{{ item }}</button>
    </nav>
    <main class="workspace">
      <div class="intro">
        <h1>{{ page }}</h1>
        <div v-if="activeEngine" class="engine-active"><span class="dot ready"></span> {{ activeEngine }}</div>
      </div>
      <div v-if="workerError" class="error-box" role="alert">
        {{ workerError }}
        <button class="button" type="button" @click="launch">重启进程</button>
      </div>

      <div v-if="page === '总览'" class="overview">
        <section class="card preview">
          <div class="preview-toolbar">
            <h2>NDI 预览</h2>
            <button class="button" type="button" :disabled="!ready" :aria-pressed="previewRate > 0"
              @click="setPreview(previewRate ? 0 : preferredPreview)">{{ previewRate ? '关闭预览' : '开启预览' }}</button>
            <label class="preview-label" for="preview-rate">帧率上限</label>
            <select id="preview-rate" v-model.number="preferredPreview">
              <option v-for="rate in previewRates" :key="rate" :value="rate">{{ rate }} FPS</option>
            </select>
          </div>
          <div class="video-surface">
            <img v-if="previewImage" :src="previewImage" alt="当前 NDI 推理预览" />
            <p v-else>{{ previewRate ? '等待预览画面' : '预览已关闭' }}</p>
          </div>
          <dl class="video-metrics">
            <div><dt>分辨率</dt><dd>{{ resolution }}</dd></div>
            <div><dt>输入帧率</dt><dd>{{ format(stream?.fps) }} <span>FPS</span></dd></div>
            <div><dt>SDK 丢帧</dt><dd>{{ stream?.dropped ?? '—' }}</dd></div>
            <div><dt>预览输出</dt><dd>{{ previewFrames }} <span>帧</span></dd></div>
          </dl>
        </section>
        <div class="configuration">
          <section class="card">
            <h2>连接状态</h2>
            <dl class="state-list">
              <div><dt>设备 A</dt><dd>{{ streamLabel }}</dd></div>
              <div><dt>推理</dt><dd>{{ modelStatus }}</dd></div>
              <div><dt>KMBox</dt><dd>{{ kmBox?.monitor ?? '监控已关闭' }}</dd></div>
            </dl>
            <button class="button" type="button" @click="page = '设备连接'">设备连接</button>
          </section>
          <section class="card">
            <h2>{{ dryRun.mode === 'real' ? '真实跟随' : 'Dry Run' }}</h2>
            <p class="status" :class="{ fault: dryRun.state === 'Fault' }">{{ stateNames[dryRun.state] }} · {{ dryRun.reason }}</p>
            <div class="readout">{{ candidate ? candidate.relative.join(' , ') : '—' }}</div>
            <p class="status">{{ dryRun.profile_name || '等待启用方案的热键' }} · {{ dryRun.mode === 'real' ? '计算位移' : '只读候选位移' }} dx / dy</p>
            <p v-if="dryRun.hold_confirmed" class="status">设备 A 持键确认有效</p>
            <p v-if="dryRun.event_hold_age_ms != null" class="status">按上次事件保持 · 距按键报文 {{ (dryRun.event_hold_age_ms / 1000).toFixed(1) }} 秒</p>
            <div class="button-group">
              <button class="button primary" type="button" :disabled="!ready || dryRun.armed" @click="setDryRun(true)">启用</button>
              <button class="button" type="button" :disabled="!dryRun.armed" @click="setDryRun(false)">停止</button>
            </div>
            <p v-if="dryError || profileError" class="status fault" role="alert">{{ dryError || profileError }}</p>
          </section>
        </div>
        <section class="card performance">
          <h2>实时性能</h2>
          <dl class="performance-metrics">
            <div><dt>推理帧率</dt><dd>{{ format(inference?.inference_fps ?? inference?.fps) }} <span>FPS</span></dd></div>
            <div><dt>模型平均延迟</dt><dd>{{ format(modelAverage, 2) }} <span>ms</span></dd></div>
            <div><dt>接收至坐标平均</dt><dd>{{ format(inference?.receiver_avg, 2) }} <span>ms</span></dd></div>
            <div><dt>接收至坐标 P95</dt><dd>{{ format(inference?.receiver_p95, 2) }} <span>ms</span></dd></div>
            <div><dt>接收至坐标 P99</dt><dd>{{ format(inference?.receiver_p99, 2) }} <span>ms</span></dd></div>
            <div><dt>NDI 等帧</dt><dd>{{ format(stream?.ndi_capture_wait_avg, 2) }} <span>ms</span></dd></div>
            <div><dt>NDI 帧复制</dt><dd>{{ format(stream?.ndi_copy_avg, 2) }} <span>ms</span></dd></div>
            <div><dt>原始帧队列等待</dt><dd>{{ format(inference?.raw_wait_avg, 2) }} <span>ms</span></dd></div>
            <div><dt>预处理</dt><dd>{{ format(inference?.preprocess_avg, 2) }} <span>ms</span></dd></div>
            <div><dt>推理队列等待</dt><dd>{{ format(inference?.prepared_wait_avg, 2) }} <span>ms</span></dd></div>
            <div><dt>目标选择与位移计算</dt><dd>{{ format(candidate?.decision_ms, 3) }} <span>ms</span></dd></div>
            <div><dt>整卡显存</dt><dd>{{ gpu?.used_mib ?? '—' }} <span>MiB</span></dd></div>
          </dl>
        </section>
      </div>

      <div v-else-if="page === '设备连接'" class="settings-grid">
        <section class="card">
          <h2>NDI 连接</h2>
          <label for="source">视频源</label>
          <div class="input-group">
            <select id="source" v-model="sourceSelection" :disabled="!sourceOptions.length">
              <option v-if="!sourceOptions.length" value="">无可用视频源</option>
              <option v-for="source in sourceOptions" :key="source.key" :value="source.key">
                {{ source.name }}
              </option>
            </select>
            <button class="button" type="button" :disabled="!ready || discovering" @click="discover">
              {{ discovering ? '发现中' : '发现' }}
            </button>
          </div>
          <div class="button-group">
            <button class="button primary" type="button" :disabled="!ready || !selectedSource" @click="changeStream(selectedSource)">接收视频</button>
            <button class="button" type="button" :disabled="!ready || streamState === 'idle'" @click="changeStream(null)">断开</button>
          </div>
          <p class="status" :class="{ fault: streamState === 'error' || streamState === 'lost' }" aria-live="polite">{{ streamLabel }}</p>
        </section>
        <section class="card">
          <h2>KMBox NET 只读监控</h2>
          <div class="form-fields">
            <label>盒子 IPv4<input v-model.trim="boxAddress" placeholder="192.168.x.x" /></label>
            <label>命令端口<input v-model.number="boxPort" type="number" min="1" max="65535" /></label>
            <label>UUID<input v-model.trim="boxUuid" maxlength="11" autocomplete="off" /></label>
            <label>本机监控 UDP 端口<input v-model.number="monitorPort" type="number" min="1024" max="49151" /></label>
          </div>
          <div class="button-group">
            <button class="button primary" type="button" :disabled="!ready || !boxAddress || !boxPort || !boxUuid"
              @click="monitor({ cmd: 'connect', ip: boxAddress, port: Number(boxPort), uuid: boxUuid })">连接盒子</button>
            <button class="button" type="button" :disabled="!kmBox?.connected || kmBox?.monitoring"
              @click="monitor({ cmd: 'monitor', port: Number(monitorPort) })">开启监控</button>
            <button class="button" type="button" :disabled="!kmBox?.monitoring" @click="monitor({ cmd: 'stop_monitor' })">关闭监控</button>
            <button class="button" type="button" :disabled="!kmBox?.connected" @click="monitor({ cmd: 'disconnect' })">断开</button>
          </div>
          <p class="status">{{ kmBox?.connection ?? '未连接' }} · {{ kmBox?.monitor ?? '监控已关闭' }} · 有效报文 {{ kmBox?.reports ?? 0 }}</p>
          <dl class="state-list">
            <div><dt>最近物理报文</dt><dd>{{ kmBox && kmBox.age_ms >= 0 ? `${kmBox.age_ms} ms 前` : '—' }}</dd></div>
            <div><dt>报文间隔 / 忽略</dt><dd>{{ kmBox?.longest_gap_ms ?? '—' }} ms / {{ kmBox?.ignored_reports ?? '—' }} 包</dd></div>
            <div><dt>最近静止持键段</dt><dd>{{ kmBox?.stationary_reports ?? '—' }} 包 · {{ kmBox?.stationary_ms ?? '—' }} ms</dd></div>
            <div><dt>静止持键最大间隔</dt><dd>{{ kmBox?.stationary_max_gap_ms ?? '—' }} ms</dd></div>
          </dl>
          <div class="key-list" aria-label="物理按键状态">
            <div v-for="(key, index) in pointerKeys" :key="key" class="key-chip"
              :class="{ down: kmBox && kmBox.buttons >= 0 && (kmBox.buttons & (1 << index)) }">
              {{ key }} · {{ !kmBox || kmBox.buttons < 0 ? '未知' : (kmBox.buttons & (1 << index)) ? '按下' : '松开' }}
            </div>
          </div>
          <p v-if="boxError" class="status fault" role="alert">{{ boxError }}</p>
          <h2>设备 A 按键确认</h2>
          <label>配对信息<input v-model.trim="inputPairing" type="password" autocomplete="off" /></label>
          <div class="button-group">
            <button class="button" type="button" :disabled="!kmBox?.monitoring || !inputPairing"
              @click="monitor({ cmd: 'confirm_input', pairing: inputPairing })">开始确认</button>
            <button class="button" type="button" :disabled="!kmBox?.connected"
              @click="monitor({ cmd: 'stop_confirmation' })">停止确认</button>
          </div>
          <p class="status">在设备 A 运行同包 VisionButtonStatus.exe。确认链路仅延续已触发的按住式 Dry Run。</p>
        </section>
      </div>

      <div v-else-if="page === '推理设置'" class="settings-grid">
        <section class="card">
          <h2>推理设置</h2>
          <label for="model-engine">推理引擎</label>
          <select id="model-engine" v-model="requestedEngine" :disabled="!ready || loadingModel" @change="loadModel">
            <option v-for="engine in engineOptions" :key="engine" :disabled="!providers.includes(engine)" :value="engine">
              {{ engine }}
            </option>
          </select>
          <label for="model-file">模型文件</label>
          <div class="input-group">
            <select id="model-file" v-model="selectedModel" :disabled="!ready || loadingModel || !modelFiles.length" @change="loadModel">
              <option v-if="!modelFiles.length" :value="selectedModel">{{ selectedModel }}</option>
              <option v-for="file in modelFiles" :key="file" :value="file">{{ file }}</option>
            </select>
            <button class="button" type="button" :disabled="!ready || loadingModel" @click="refreshModels">刷新</button>
          </div>
          <div class="button-group">
            <button class="button primary" type="button" :disabled="!ready || loadingModel || !providers.includes(requestedEngine)" @click="loadModel">
              {{ loadingModel ? '加载中' : '加载模型' }}
            </button>
            <button class="button" type="button" :disabled="!ready || cudaChecking" @click="checkCuda">CUDA 自检</button>
          </div>
          <p class="status" :class="{ fault: !loadingModel && !activeEngine && modelStatus !== '模型未加载' }" aria-live="polite">{{ modelStatus }}</p>
          <p v-if="cudaStatus" class="status" aria-live="polite">{{ cudaStatus }}</p>
        </section>
        <section class="card">
          <h2>实际后端与缓存</h2>
          <dl class="state-list">
            <div><dt>后端</dt><dd>{{ activeEngine || '—' }} {{ backend?.version }}</dd></div>
            <div><dt>精度</dt><dd>{{ backend?.precision ?? '—' }}</dd></div>
            <div><dt>引擎缓存</dt><dd>{{ backend?.cache === 'hit' ? '已命中' : backend?.cache === 'built' ? '已构建' : backend?.cache ?? '—' }}</dd></div>
            <div><dt>RTX 运行时缓存</dt><dd>{{ backend?.jit_cache ?? '—' }}</dd></div>
            <div><dt>模型 P95 / P99</dt><dd>{{ format(inference?.model_p95, 2) }} / {{ format(inference?.model_p99, 2) }} ms</dd></div>
            <div><dt>预处理前等待</dt><dd>{{ format(inference?.before_preprocess_avg, 2) }} ms</dd></div>
            <div><dt>预处理</dt><dd>{{ format(inference?.preprocess_avg, 2) }} ms</dd></div>
            <div><dt>等待推理</dt><dd>{{ format(inference?.prepared_wait_avg, 2) }} ms</dd></div>
          </dl>
        </section>
      </div>

      <div v-else-if="page === '热键管理'" class="settings-grid">
        <section class="card">
          <div class="section-toolbar"><h2>热键方案</h2><button class="button" @click="addProfile">新建方案</button></div>
          <label>编辑方案
            <select v-model="selection"><option v-for="entry in profiles" :key="entry.id" :value="entry.id">
              {{ entry.name }} · {{ entry.enabled ? '启用' : '停用' }}
            </option></select>
          </label>
          <label>名称<input v-model.trim="profile.name" maxlength="36" /></label>
          <label class="key-chip"><input v-model="profile.enabled" type="checkbox" />启用此方案</label>
          <button class="button" :disabled="profiles.length < 2" @click="deleteProfile">删除当前方案</button>
          <p class="status">同时匹配多个方案时，优先触发按键更多的组合。同等键数按列表顺序。配置自动保存。</p>
          <p v-if="conflicts.length" class="status fault" role="alert">按键重复：{{ conflicts.map(entry => entry.name).join('、') }}</p>
          <p v-else-if="profileError" class="status fault" role="alert">{{ profileError }}</p>
          <div class="button-group">
            <button class="button" type="button" :disabled="configBusy" @click="exportConfiguration">导出配置</button>
            <button class="button" type="button" :disabled="configBusy || realArming" @click="backupInput?.click()">导入配置</button>
            <button class="button" type="button" :disabled="configBusy || realArming || !previousBackup"
              @click="changeConfiguration(async () => JSON.stringify(previousBackup), '已恢复上次导入前的配置')">恢复上次配置</button>
            <input ref="backupInput" type="file" accept=".json,application/json" hidden @change="importConfiguration" />
          </div>
          <p v-if="configStatus" class="status" :class="{ fault: configFailed }" aria-live="polite">{{ configStatus }}</p>
        </section>
        <section class="card">
          <h2>物理按键与触发方式</h2>
          <div class="key-list">
            <label v-for="(key, index) in pointerKeys" :key="key" class="key-chip">
              <input type="checkbox" :checked="!!(profile.mask & (1 << index))"
                @change="pressKey(index, ($event.target as HTMLInputElement).checked)" />{{ key }}
            </label>
          </div>
          <label>触发方式
            <select v-model="profile.mode"><option value="hold">按住执行</option>
              <option value="toggle" :disabled="(profile.mask & 3) === 3">按键切换</option>
            </select>
          </label>
          <p v-if="(profile.mask & 3) === 3" class="status">左键和右键须同时按住，松开任一键即结束触发。</p>
          <label v-if="profile.mode === 'hold'" class="key-chip"><input v-model="profile.eventPreview" type="checkbox" />Dry Run 按事件保持候选</label>
          <p v-if="profile.mode === 'hold' && profile.eventPreview" class="status">启动后条件就绪即等待热键，松键后继续等待下一次按下。若松键报文丢失，按 F8 停止。</p>
        </section>
      </div>

      <div v-else-if="page === '目标识别'" class="settings-grid">
        <section class="card">
          <h2>启用的 YOLO 类别</h2>
          <div class="key-list">
            <label v-for="id in classIds" :key="id" class="key-chip">
              <input v-model="enabledClasses" type="checkbox" :value="id" />类别 {{ id }}
            </label>
          </div>
          <p class="status">当前推理帧检测框 {{ inference?.count ?? '—' }}</p>
        </section>
        <section class="card">
          <h2>{{ profile.name }} · 筛选与优先级</h2>
          <label>类别范围
            <select v-model="profile.scope"><option value="all">全部已启用类别</option>
              <option value="single">单类别</option><option value="custom">自定义多类别</option>
            </select>
          </label>
          <label v-if="profile.scope === 'single'">选择类别
            <select v-model="profile.single"><option v-for="id in classIds" :key="id" :value="id">类别 {{ id }}</option></select>
          </label>
          <div v-if="profile.scope === 'custom'" class="key-list">
            <label v-for="id in classIds" :key="id" class="key-chip">
              <input v-model="profile.classes" type="checkbox" :value="id" />类别 {{ id }}
            </label>
          </div>
          <label>目标优先级
            <select v-model="profile.priority"><option value="closest">距画面中心最近</option>
              <option value="confidence">置信度最高</option><option value="largest">目标框最大</option>
            </select>
          </label>
          <div class="form-fields">
            <label>最低置信度 (%)<input v-model.number="profile.confidence" type="number" min="55" max="100" /></label>
            <label>最大有效半径 (%)<input v-model.number="profile.rangePercent" type="number" min="1" max="100"
              @change="profile.dynamicRange !== undefined && (profile.dynamicRange = Math.min(profile.dynamicRange, profile.rangePercent))" /></label>
            <label class="key-chip"><input type="checkbox" :checked="profile.dynamicRange !== undefined"
              @change="profile.dynamicRange = ($event.target as HTMLInputElement).checked ? Math.min(50, profile.rangePercent) : undefined" />按目标速度扩展有效范围</label>
            <label v-if="profile.dynamicRange !== undefined">动态起始半径 (%)
              <input v-model.number="profile.dynamicRange" type="number" min="1" :max="profile.rangePercent" />
            </label>
            <label>切换目标确认帧数<input v-model.number="profile.switchConfirmFrames" type="number" min="1" max="8" /></label>
            <label>Dry Run 遮挡等待（ms）<input v-model.number="profile.lostWaitMs" type="number" min="0" max="500" /></label>
            <label>锁定点水平偏移 (%)<input v-model.number="profile.anchorX" type="number" min="-50" max="50" /></label>
            <label>锁定点垂直偏移 (%)<input v-model.number="profile.anchorY" type="number" min="-50" max="50" /></label>
            <label>参考点 X (%)<input v-model.number="profile.referenceX" type="number" min="0" max="100" /></label>
            <label>参考点 Y (%)<input v-model.number="profile.referenceY" type="number" min="0" max="100" /></label>
          </div>
          <p class="status">偏移按检测框大小换算。有效半径 100% 等于画面短边的一半。置信度最低 55% 为当前检测后端的输出阈值。</p>
          <details>
            <summary>各类别置信度、优先级和偏移</summary>
            <p class="status">阈值留空时用方案阈值。初次选目标时，优先级数字越大越优先。</p>
            <div v-for="id in classIds" :key="id" class="form-fields">
              <label>类别 {{ id }} 阈值 (%)
                <input v-if="profile.classRules[id]" v-model.number="profile.classRules[id].threshold" type="number" min="55" max="100" />
              </label>
              <label>类别 {{ id }} 优先级
                <input v-if="profile.classRules[id]" v-model.number="profile.classRules[id].priority" type="number" min="-100" max="100" />
              </label>
              <template v-if="profile.classRules[id]">
                <label class="key-chip"><input type="checkbox" :checked="!!profile.classRules[id].anchor"
                  @change="profile.classRules[id].anchor = ($event.target as HTMLInputElement).checked ? [profile.anchorX, profile.anchorY] : undefined" />类别 {{ id }} 自定义偏移</label>
                <template v-if="profile.classRules[id].anchor">
                  <label>X 偏移 (%)<input v-model.number="profile.classRules[id].anchor![0]" type="number" min="-50" max="50" /></label>
                  <label>Y 偏移 (%)<input v-model.number="profile.classRules[id].anchor![1]" type="number" min="-50" max="50" /></label>
                </template>
              </template>
            </div>
          </details>
        </section>
      </div>

      <div v-else-if="page === '鼠标控制'" class="settings-grid">
        <section class="card">
          <h2>跟随控制</h2>
          <label>运行模式
            <select v-model="controlMode">
              <option value="dry_run">Dry Run</option>
              <option value="calibration">单步校准</option>
              <option value="real">Real Auto Tracking</option>
            </select>
          </label>
          <div v-if="controlMode === 'real'" class="form-fields">
            <label class="key-chip"><input v-model="dryRunPassed" type="checkbox" :disabled="realArming || dryRun.mode === 'real'" />已完成现场 Dry Run</label>
            <label class="key-chip"><input v-model="stepPassed" type="checkbox" :disabled="realArming || dryRun.mode === 'real'" />已完成单步校准及坐标保存</label>
            <label class="key-chip"><input v-model="fullPassed" type="checkbox" :disabled="!dryRunPassed || !stepPassed || realArming || dryRun.mode === 'real'" />已通过单目标跟随、移动目标及停止项</label>
            <p class="status">{{ fullPassed ? '多方案跟随。可使用完成校准的按住式热键。' : '单目标低速验收。当前方案须为左右组合键并且仅选一个类别。' }}</p>
            <p class="status">真实跟随期间请保持此窗口焦点。按 F8 急停。</p>
            <button class="button danger" type="button" :disabled="!ready || !dryRunPassed || !stepPassed || realArming || dryRun.mode === 'real'" @click="armReal">武装真实跟随</button>
            <details>
              <summary>现场停止验收</summary>
              <ol>
                <li>在普通桌面测试场景完成单步移动、坐标校准和 Dry Run，再武装单目标低速跟随。</li>
                <li>分别只按左键和右键，再同时按住两键。分别验证松开左键和松开右键时停止。</li>
                <li>独立验证 F8、目标移出画面、NDI 断流和 KMBox 断连。每项恢复后确认需要重新武装。</li>
                <li>导出运行日志，对照停止时间、锁止原因及清队列结果。</li>
              </ol>
            </details>
          </div>
          <div class="form-fields">
            <label>水平校准比例<input v-model.number="profile.gainX" type="number" min="0.01" max="10" step="0.01" /></label>
            <label>垂直校准比例<input v-model.number="profile.gainY" type="number" min="0.01" max="10" step="0.01" /></label>
            <label>真实跟随 X 增益<input v-model.number="profile.trackingGain[0]" type="number" min="0.01" max="2" step="0.01" /></label>
            <label>真实跟随 Y 增益<input v-model.number="profile.trackingGain[1]" type="number" min="0.01" max="2" step="0.01" /></label>
            <label>移动算法<select v-model="profile.motion.algorithm"><option value="p">P</option><option value="pd">PD</option></select></label>
            <template v-if="profile.motion.algorithm === 'pd'">
              <label>X 微分时间（ms）<input v-model.number="profile.motion.derivativeMs[0]" type="number" min="0" max="120" /></label>
              <label>Y 微分时间（ms）<input v-model.number="profile.motion.derivativeMs[1]" type="number" min="0" max="120" /></label>
            </template>
            <label>预测时长（ms，0 关闭）<input v-model.number="profile.motion.predictionMs" type="number" min="0" max="80" /></label>
            <label>近目标减速半径（px，0 关闭）<input v-model.number="profile.motion.slowRadiusPx" type="number" min="0" max="160" /></label>
            <label class="key-chip"><input v-model="profile.motion.adaptive" type="checkbox" />自适应平滑</label>
            <label>单轴上限<input v-model.number="profile.maxStep" type="number" min="1" max="24" /></label>
            <label>死区（NDI 像素）<input v-model.number="profile.deadZonePx" type="number" min="0" max="96" step="0.1" /></label>
            <label>平滑半衰期（ms）<input v-model.number="profile.smoothHalfLifeMs" type="number" min="0" max="150" step="0.1" /></label>
            <label>真实跟随速度上限（单位/秒）<input v-model.number="profile.maxSpeed" type="number" min="90" max="1200" /></label>
            <label>真实跟随加速度上限（单位/秒²）<input v-model.number="profile.acceleration" type="number" min="1" max="20000" /></label>
            <label class="key-chip"><input type="checkbox" :checked="!!profile.deadZoneAxes"
              @change="profile.deadZoneAxes = ($event.target as HTMLInputElement).checked ? [profile.deadZonePx, profile.deadZonePx] : undefined" />分开设置 X/Y 死区</label>
            <template v-if="profile.deadZoneAxes">
              <label>水平死区（NDI 像素）<input v-model.number="profile.deadZoneAxes[0]" type="number" min="0" max="96" step="0.1" /></label>
              <label>垂直死区（NDI 像素）<input v-model.number="profile.deadZoneAxes[1]" type="number" min="0" max="96" step="0.1" /></label>
            </template>
          </div>
          <div class="key-list">
            <label class="key-chip"><input v-model="profile.invertX" type="checkbox" />反转水平轴</label>
            <label class="key-chip"><input v-model="profile.invertY" type="checkbox" />反转垂直轴</label>
          </div>
          <p class="status">参考点按 NDI 画面百分比配置。半衰期填 0 可关闭平滑。</p>
          <div class="button-group">
            <button v-if="controlMode === 'dry_run'" class="button primary" :disabled="!ready || dryRun.armed" @click="setDryRun(true)">启用 Dry Run</button>
            <button class="button" :disabled="!dryRun.armed" @click="setDryRun(false)">停止</button>
          </div>
          <p class="status" :class="{ fault: dryError || profileError || dryRun.state === 'Fault' }">{{ dryError || profileError || dryRun.reason }}</p>
          <p v-if="dryRun.last_stop" class="status">上次停止 · {{ new Date(dryRun.last_stop.at_ms).toLocaleTimeString() }} · {{ dryRun.last_stop.reason }}</p>
        </section>
        <section class="card">
          <h2>{{ dryRun.mode === 'real' ? '真实跟随计算结果' : 'Dry Run 计算结果' }}</h2>
          <div v-if="rangeDiagram" class="range-diagram" role="img" aria-label="NDI 有效范围示意"
            :style="{ aspectRatio: rangeDiagram.aspectRatio }">
            <span class="range-circle" :style="rangeDiagram.ring"></span>
            <span v-if="rangeDiagram.anchor" class="range-anchor" :style="rangeDiagram.anchor"></span>
          </div>
          <p v-if="dryRun.event_hold_age_ms != null" class="status">按上次事件保持 · 距按键报文 {{ (dryRun.event_hold_age_ms / 1000).toFixed(1) }} 秒</p>
          <div class="readout">{{ candidate ? candidate.relative.join(' , ') : '—' }} <small>dx / dy</small></div>
          <dl class="state-list">
            <div><dt>触发方案</dt><dd>{{ dryRun.profile_name ?? '—' }}</dd></div>
            <div><dt>检测类别</dt><dd>{{ candidate ? `${candidate.class} · ${format(candidate.score * 100)}%` : '—' }}</dd></div>
            <div><dt>选择原因</dt><dd>{{ candidate?.selection_reason ?? '—' }}</dd></div>
            <div><dt>锁定编号</dt><dd>{{ candidate?.lock_id ?? '—' }}</dd></div>
            <div><dt>Track ID</dt><dd>{{ candidate?.track_id ?? '—' }}</dd></div>
            <div><dt>目标速度 X/Y</dt><dd>{{ candidate ? candidate.target_velocity_px_s.map(v => format(v)).join(' / ') + ' px/s' : '—' }}</dd></div>
            <div><dt>当前有效半径</dt><dd>{{ candidate ? format(candidate.range_px) + ' px' : '—' }}</dd></div>
            <div><dt>已调度 / 已过期</dt><dd>{{ dryRun.scheduled ?? 0 }} / {{ dryRun.expired ?? 0 }}</dd></div>
            <div><dt>最近一次队列等待</dt><dd>{{ format(dryRun.queue_ms, 2) }} ms</dd></div>
            <div><dt>检测框中心</dt><dd>{{ candidate?.center.map(v => format(v)).join(', ') ?? '—' }}</dd></div>
            <div><dt>锁定点</dt><dd>{{ candidate?.anchor.map(v => format(v)).join(', ') ?? '—' }}</dd></div>
            <div><dt>参考点</dt><dd>{{ candidate?.reference.join(', ') ?? '—' }}</dd></div>
            <div><dt>归一化坐标</dt><dd>{{ candidate?.normalized.map(v => format(v, 3)).join(', ') ?? '—' }}</dd></div>
            <div><dt>参考偏移</dt><dd>{{ candidate?.offset.map(v => format(v)).join(', ') ?? '—' }}</dd></div>
            <div><dt>实际换算比例 X/Y</dt><dd>{{ candidate?.signed_scale.map(v => format(v, 6)).join(' / ') ?? '—' }}</dd></div>
            <div><dt>选择与计算耗时</dt><dd>{{ candidate ? `${format(candidate.decision_ms, 3)} ms` : '—' }}</dd></div>
            <div><dt>接收到候选位移</dt><dd>{{ candidate ? `${candidate.latency_ms} ms` : '—' }}</dd></div>
          </dl>
        </section>
        <section class="card">
          <h2>应用手动校准测量</h2>
          <p class="status">在设备 A 的普通桌面完成 +10 水平与 +10 垂直单次测试。验收窗口提供五键状态和 F8 急停。</p>
          <button v-if="controlMode === 'calibration'" class="button" type="button" @click="openKmBoxAcceptance">打开单步校准</button>
          <p v-if="boxError" class="status fault" role="alert">{{ boxError }}</p>
          <p class="status">测量时使用 NDI 图像坐标，完成后填写有符号位移。</p>
          <div class="form-fields">
            <label>水平实测像素<input v-model.number="observedX" type="number" step="any" /></label>
            <label>垂直实测像素<input v-model.number="observedY" type="number" step="any" /></label>
            <label>设备 A 桌面宽（像素）<input v-model.number="desktopWidth" type="number" min="1" /></label>
            <label>设备 A 桌面高（像素）<input v-model.number="desktopHeight" type="number" min="1" /></label>
          </div>
          <button class="button" :disabled="observedX === '' || observedY === ''" @click="applyCalibration">换算并保存到当前方案</button>
          <p v-if="calibrationStatus" class="status" :class="{ fault: calibrationFailed }" aria-live="polite">{{ calibrationStatus }}</p>
        </section>
      </div>

      <section v-else class="card activity">
        <div class="section-toolbar">
          <h2>运行日志</h2>
          <button class="button" type="button" :aria-pressed="deviceAOperation" @click="markDeviceAOperation">
            {{ deviceAOperation ? '记录设备 A 操作结束' : '记录设备 A 操作开始' }}
          </button>
          <button class="button" type="button" :disabled="!logs.length" @click="exportLog">导出日志</button>
        </div>
        <p v-if="logPath" class="status" aria-live="polite">日志已保存到 {{ logPath }}</p>
        <p v-if="logError" class="status fault" role="alert">{{ logError }}</p>
        <p v-if="!logs.length" class="status">暂无运行日志</p>
        <ul v-else class="log-lines" aria-label="最近 200 条运行事件">
          <li v-for="(entry, index) in logs.slice(-200).reverse()" :key="index" :class="{ fault: entry.level === 'error' }">
            <time :datetime="entry.at.toISOString()">{{ entry.at.toLocaleTimeString('zh-CN', { hour12: false }) }}</time>
            <span>{{ entry.text }}</span>
          </li>
        </ul>
      </section>
    </main>
  </div>
</template>
