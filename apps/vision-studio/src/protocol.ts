export const engines = ['Auto', 'CUDA', 'TensorRT', 'TensorRT for RTX', 'CPU'] as const
export type Engine = typeof engines[number]

export type Command =
  | { cmd: 'discover' | 'cuda_check' }
  | { cmd: 'stream'; source: number; name: string; url: string }
  | { cmd: 'model'; model: number; provider: Engine; filename?: string }
  | { cmd: 'preview'; fps: number; preview: number }

export type MonitorCommand =
  | { cmd: 'connect'; ip: string; port: number; uuid: string }
  | { cmd: 'monitor'; port: number }
  | { cmd: 'stop_monitor' | 'disconnect' }
  | { cmd: 'confirm_input'; pairing: string }
  | { cmd: 'stop_confirmation' }

export type Stream = {
  kind: 'stream'
  source: number
  error?: string
  width?: number
  height?: number
  fps?: number
  dropped?: number
  replaced?: number
  preview_frames?: number
  ndi_capture_wait_avg?: number | null
  ndi_copy_avg?: number | null
}

export type Inference = {
  kind: 'inference'
  source: number
  model: number
  runs: number
  count: number
  average?: number
  warm_samples?: number
  model_avg?: number
  model_p95?: number
  model_p99?: number
  receiver_avg?: number
  receiver_p95?: number
  receiver_p99?: number
  before_preprocess_avg?: number
  preprocess_avg?: number
  raw_wait_avg?: number
  prepared_wait_avg?: number
  inference_fps?: number
}

export type Candidate = {
  profile_id: number
  class: number
  score: number
  bounds: [number, number, number, number]
  center: [number, number]
  anchor: [number, number]
  reference: [number, number]
  normalized: [number, number]
  offset: [number, number]
  signed_scale: [number, number]
  relative: [number, number]
  dimensions: [number, number]
  sequence: number
  lock_id: number
  track_id: number
  target_velocity_px_s: [number, number]
  range_px: number
  latency_ms: number
  decision_ms: number
  selection_ms: number
  movement_ms: number
  selection_reason: string
}

export type DryRun = {
  kind: 'dry_run'; state: 'Idle' | 'Ready' | 'Armed' | 'Running' | 'Stopped' | 'Fault'
  armed: boolean; active: boolean; reason: string; candidate: Candidate | null
  profile_id?: number | null; profile_name?: string | null
  mode: 'dry_run' | 'real'; real_control_available: boolean
  scheduled?: number; expired?: number
  hold_confirmed?: boolean
  event_hold_age_ms?: number | null
}
export type KmBox = {
  kind: 'kmbox'; connected: boolean; monitoring: boolean; buttons: number
  reports: number; age_ms: number; connection: string; monitor: string; halted: boolean
  report_at_ms?: number; longest_gap_ms?: number; ignored_reports?: number
  stationary_reports?: number; stationary_ms?: number; stationary_max_gap_ms?: number
}
export type RateDip = {
  kind: 'rate_dip'; metric: 'input' | 'inference'; fps: number; threshold: number; duration_ms: number | null
  frame_age_ms: number | null; raw_pending: number | null; prepared_pending: number | null
  target_sequence: number | null; target_updated_at_ms: number | null; target_age_ms: number | null
}
export type Gpu = { kind: 'gpu'; used_mib: number | null; total_mib: number | null }
export type Backend = {
  version?: string; precision?: string; cache?: string; jit_cache?: string
  previous_attempts?: { provider: string; error: string }[]
}
export type WorkerEvent =
  | Stream | Inference | DryRun | KmBox | Gpu | RateDip
  | { kind: 'ready'; providers?: Engine[]; preview_frames?: number }
  | { kind: 'worker_exit' | 'kmbox_exit' }
  | { kind: 'preview_state'; fps: number; preview: number; frames: number; error?: string }
  | { kind: 'preview'; source: number; preview: number; image: string }
  | { kind: 'stream_state'; state: 'lost' | 'resumed' | 'live'; source: number; gap_ms: number }
  | { kind: 'sources'; sources: [string, string][]; error?: string }
  | { kind: 'model'; model: number; error?: string; phase?: 'load' | 'inference'; requested?: Engine; provider?: Engine; metadata?: string; backend?: Backend }
  | { kind: 'model_loading'; model: number; status: string }
  | { kind: 'cuda_check'; status: string; details?: string[] }
  | { kind: 'kmbox_error'; error: string }
  | { kind: 'move_receipt'; id: number; target: number; class: number; dx: number; dy: number;
      status: string; exchange_ms: number; dispatch_ms: number; receiver_ms: number }
  | { kind: 'log'; text: string }
