namespace VisionHardwareCheck
{
    internal static class DryRunGate
    {
        internal static string BlockReason(KmBox.Snapshot km, bool streamValid, bool inferenceValid)
        {
            if (km.Halted) return "紧急停止";
            if (!km.Connected) return "KMBox 未连接";
            if (!km.Monitoring || km.Buttons < 0) return "物理按键监控未就绪";
            if (!streamValid) return "NDI 无新帧";
            if (!inferenceValid) return "推理未就绪或已过期";
            return null;
        }

        internal static bool IsOpen(bool armed, string blockReason, KmBox.Snapshot km)
        {
            return armed && blockReason == null && km.Combo && (km.Buttons & KmBox.LeftRight) == KmBox.LeftRight;
        }
    }
}
