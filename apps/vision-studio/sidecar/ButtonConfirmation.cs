using System;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Threading;

internal static class ButtonConfirmation
{
    internal const int Port = 24177;
    const byte Request = 1, Reply = 2;
    const int NonceSize = 16, PayloadSize = 18, PacketSize = 50;

    internal static byte[] NewKey()
    {
        byte[] key = new byte[16];
        using (var random = RandomNumberGenerator.Create()) random.GetBytes(key);
        return key;
    }

    internal static byte[] ParseKey(string text)
    {
        if (text.Length != 32) throw new ArgumentException("请输入设备 A 显示的 32 位配对码");
        byte[] key = new byte[16];
        for (int i = 0; i < key.Length; i++) key[i] = Convert.ToByte(text.Substring(i * 2, 2), 16);
        return key;
    }

    static byte[] Sign(HMACSHA256 hmac, byte kind, byte[] nonce, int offset, byte mask)
    {
        byte[] result = new byte[PacketSize];
        result[0] = kind;
        Buffer.BlockCopy(nonce, offset, result, 1, NonceSize);
        result[17] = mask;
        Buffer.BlockCopy(hmac.ComputeHash(result, 0, PayloadSize), 0, result, PayloadSize, 32);
        return result;
    }

    static bool Valid(HMACSHA256 hmac, byte[] packet, int length, byte kind)
    {
        if (length != PacketSize || packet[0] != kind || packet[17] > 31 || (kind == Request && packet[17] != 0)) return false;
        byte[] digest = hmac.ComputeHash(packet, 0, PayloadSize);
        int difference = 0;
        for (int i = 0; i < digest.Length; i++) difference |= digest[i] ^ packet[PayloadSize + i];
        return difference == 0;
    }

    internal static void Respond(UdpClient listener, byte[] key, Func<byte> readButtons, Action onRequest, CancellationToken stop)
    {
        using (var auth = new HMACSHA256(key))
        {
            IPEndPoint sender = new IPEndPoint(IPAddress.Any, 0);
            while (!stop.IsCancellationRequested)
            {
                if (!listener.Client.Poll(25000, SelectMode.SelectRead)) continue;
                byte[] request = listener.Receive(ref sender);
                if (!Valid(auth, request, request.Length, Request)) continue;
                byte[] reply = Sign(auth, Reply, request, 1, readButtons());
                listener.Send(reply, reply.Length, sender);
                onRequest();
            }
        }
    }

    internal static void Poll(IPAddress address, byte[] key, Action<byte,long> report, CancellationToken stop)
    {
        using (var client = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp))
        using (var auth = new HMACSHA256(key))
        using (var random = RandomNumberGenerator.Create())
        {
            client.Connect(new IPEndPoint(address, Port));
            byte[] nonce = new byte[NonceSize];
            byte[] buffer = new byte[512];
            while (!stop.IsCancellationRequested)
            {
                try
                {
                    random.GetBytes(nonce);
                    byte[] query = Sign(auth, Request, nonce, 0, 0);
                    long startedAt = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                    var deadline = Stopwatch.StartNew();
                    client.Send(query);
                    while (!stop.IsCancellationRequested && deadline.ElapsedMilliseconds < 45)
                    {
                        if (!client.Poll(5000, SelectMode.SelectRead)) continue;
                        int length = client.Receive(buffer);
                        if (!Valid(auth, buffer, length, Reply)) continue;
                        int difference = 0;
                        for (int i = 0; i < nonce.Length; i++) difference |= nonce[i] ^ buffer[i+1];
                        if (difference != 0) continue;
                        report(buffer[17], startedAt);
                        break;
                    }
                }
                catch (SocketException) { /* Missing responses expire on the controller. */ }
                stop.WaitHandle.WaitOne(30);
            }
        }
    }
}
