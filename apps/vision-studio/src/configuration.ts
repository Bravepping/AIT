import { engines, type Engine } from './protocol'

export const classIds = [0, 1, 2, 3, 4, 5, 6]

export type Profile = {
  id: number; name: string; mask: number; mode: 'hold' | 'toggle'; scope: 'all' | 'single' | 'custom';
  single: number; classes: number[]; priority: 'closest' | 'confidence' | 'largest';
  enabled: boolean; confidence: number; anchorX: number; anchorY: number; rangePercent: number;
  gainX: number; gainY: number; invertX: boolean; invertY: boolean; maxStep: number
  deadZonePx: number; smoothHalfLifeMs: number; switchConfirmFrames: number
  calibration?: { observedX: number; observedY: number; source?: string; device?: string;
    desktop?: [number, number]; crop?: [number, number] }
  eventPreview?: boolean
  classRules: Record<number, { threshold: number | ''; priority: number; anchor?: [number, number] }>
  deadZoneAxes?: [number, number]
  maxSpeed: number
  acceleration: number
  trackingGain: [number, number]
  referenceX: number
  referenceY: number
  lostWaitMs: number
  dynamicRange?: number
  motion: { algorithm: 'p' | 'pd'; derivativeMs: [number, number]; predictionMs: number; slowRadiusPx: number; adaptive: boolean }
}
export const newProfile = (id: number): Profile => ({
  id, name: id === 1 ? '左右键组合' : `方案 ${id}`, mask: 3, mode: 'hold', scope: 'all',
  single: 0, classes: [...classIds], priority: 'closest', enabled: id === 1,
  confidence: 55, anchorX: 0, anchorY: 0, rangePercent: 100,
  gainX: 1, gainY: 1, invertX: false, invertY: false, maxStep: 24,
  deadZonePx: 2, smoothHalfLifeMs: 12, switchConfirmFrames: 2,
  classRules: Object.fromEntries(classIds.map(id => [id, { threshold: '', priority: 0 }])),
  maxSpeed: 480, acceleration: 4000, trackingGain: [0.35, 0.35],
  referenceX: 50, referenceY: 50,
  lostWaitMs: 120,
  motion: { algorithm: 'p', derivativeMs: [16, 16], predictionMs: 0, slowRadiusPx: 0, adaptive: false },
})

export type Preferences = {
  boxAddress: string; boxPort: number | ''; boxUuid: string; monitorPort: number
  engine: Engine; model: string; source: string; preview: number
}

export type Backup = {
  format: 'ndi-vision-control'; version: 1; profiles: Profile[]
  enabledClasses: number[]; selection: number; preferences: Preferences
}

export function migrateProfile(entry: Profile): Profile {
  const defaults = newProfile(entry.id)
  return { ...defaults, ...entry, lostWaitMs: entry.lostWaitMs ?? 0,
    motion: { ...defaults.motion, ...entry.motion },
    classRules: { ...defaults.classRules, ...entry.classRules },
    maxStep: Math.min(24, Math.max(1, Number(entry.maxStep) || 24)) }
}

export function makeBackup(profiles: Profile[], enabledClasses: number[], selection: number, preferences: Preferences): Backup {
  return { format: 'ndi-vision-control', version: 1, profiles, enabledClasses, selection, preferences }
}

function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('配置中的对象格式有误')
  return value as Record<string, unknown>
}

function integer(value: unknown, minimum: number, maximum: number): value is number {
  return typeof value === 'number' && Number.isInteger(value) && minimum <= value && value <= maximum
}

function pair(value: unknown, item: (n: unknown) => boolean): boolean {
  return Array.isArray(value) && value.length === 2 && value.every(item)
}

function finite(value: unknown): value is number { return typeof value === 'number' && Number.isFinite(value) }

export function parseBackup(serialized: string): Backup {
  if (serialized.length > 1024 * 1024) throw new Error('配置文件超出 1 MB')
  const data = object(JSON.parse(serialized))
  if (data.format !== 'ndi-vision-control' || data.version !== 1) throw new Error('不支持此配置文件版本')
  if (!Array.isArray(data.profiles) || data.profiles.length === 0 || data.profiles.length > 64) {
    throw new Error('配置须包含 1–64 个热键方案')
  }
  const ids = new Set<number>()
  const profiles = data.profiles.map((raw) => {
    const entry = object(raw)
    const { id, mode, scope, priority, name, enabled, mask } = entry
    if (!integer(id, 1, 0xffffffff) || ids.has(id) || typeof name !== 'string' || name.trim().length === 0 ||
      name.length > 36 || typeof enabled !== 'boolean' || !integer(mask, enabled ? 1 : 0, 31) ||
      !['hold', 'toggle'].includes(String(mode)) || !['all', 'single', 'custom'].includes(String(scope)) ||
      !['closest', 'confidence', 'largest'].includes(String(priority))) throw new Error('热键方案无效或编号重复')
    ids.add(id)
    const migrated = migrateProfile(entry as Profile)
    const numbers = [migrated.confidence, migrated.anchorX, migrated.anchorY, migrated.rangePercent,
      migrated.gainX, migrated.gainY, migrated.deadZonePx, migrated.smoothHalfLifeMs, migrated.switchConfirmFrames,
      migrated.maxSpeed, migrated.acceleration, migrated.referenceX, migrated.referenceY, migrated.lostWaitMs,
      migrated.motion.predictionMs, migrated.motion.slowRadiusPx]
    if (!numbers.every(finite) || (entry.maxStep !== undefined && !integer(entry.maxStep, 1, 24)) ||
      !integer(migrated.single, 0, 6) || !Array.isArray(migrated.classes) || !migrated.classes.every((v) => integer(v, 0, 6)) ||
      !pair(migrated.trackingGain, finite) || !pair(migrated.motion.derivativeMs, finite) ||
      !['p', 'pd'].includes(migrated.motion.algorithm) || typeof migrated.motion.adaptive !== 'boolean' ||
      [migrated.invertX, migrated.invertY].some((v) => typeof v !== 'boolean') ||
      (migrated.eventPreview !== undefined && typeof migrated.eventPreview !== 'boolean') ||
      (migrated.dynamicRange !== undefined && !finite(migrated.dynamicRange)) ||
      (migrated.deadZoneAxes !== undefined && !pair(migrated.deadZoneAxes, finite))) throw new Error('方案参数格式有误')
    for (const [key, rule] of Object.entries(migrated.classRules)) {
      if (!classIds.map(String).includes(key) || !rule || !integer(rule.priority, -100, 100) ||
        (rule.threshold !== '' && !finite(rule.threshold)) ||
        (rule.anchor !== undefined && !pair(rule.anchor, finite))) throw new Error('类别参数格式有误')
    }
    const calibration = migrated.calibration
    if (calibration !== undefined && (!calibration || !finite(calibration.observedX) || !finite(calibration.observedY) ||
      (calibration.desktop !== undefined && !pair(calibration.desktop, (v) => integer(v, 1, 16384))) ||
      (calibration.crop !== undefined && !pair(calibration.crop, (v) => integer(v, 1, 8192))) ||
      (calibration.source !== undefined && typeof calibration.source !== 'string') ||
      (calibration.device !== undefined && typeof calibration.device !== 'string'))) throw new Error('校准参数格式有误')
    return migrated
  })
  const p = object(data.preferences)
  if (typeof p.boxAddress !== 'string' || p.boxAddress.length > 64 || typeof p.boxUuid !== 'string' || p.boxUuid.length > 11 ||
    (p.boxPort !== '' && !integer(p.boxPort, 1, 65535)) || !integer(p.monitorPort, 1024, 49151) ||
    !engines.includes(p.engine as Engine) || typeof p.model !== 'string' || !/^[^/\\:*?"<>|]+\.onnx$/i.test(p.model) ||
    typeof p.source !== 'string' || p.source.length > 512 || ![5, 10, 15, 30, 60].includes(Number(p.preview))) {
    throw new Error('设备或推理偏好格式有误')
  }
  if (!Array.isArray(data.enabledClasses) || !data.enabledClasses.every((v) => integer(v, 0, 6))) throw new Error('类别列表格式有误')
  const { boxAddress, boxPort, boxUuid, monitorPort, engine, model, source } = p as Preferences
  return makeBackup(profiles, [...new Set(data.enabledClasses as number[])],
    integer(data.selection, 1, 0xffffffff) && ids.has(data.selection) ? data.selection : profiles[0]!.id,
    { boxAddress, boxPort, boxUuid, monitorPort, engine, model, source, preview: Number(p.preview) })
}
