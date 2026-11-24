// SmartSearch.exe: starts the program in its own window, without a console.
// While Python is starting (it takes a few seconds) a small splash window with a running progress bar is
// shown, so the user sees that the click worked. main.py signals the named event "Local\SmartSearchReady"
// once its window has loaded the page; then the splash closes. If Python exits first, the splash says so.
// Built by scripts/build_launcher.py with the C# compiler that ships with Windows (.NET Framework 4.x).
using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Threading;
using System.Windows.Forms;

[assembly: System.Reflection.AssemblyTitle("Bilingual Text Analyzer")]
[assembly: System.Reflection.AssemblyProduct("Bilingual Text Analyzer")]

static class Launcher
{
    const string Title = "Bilingual Text Analyzer";
    const string ReadyEvent = @"Local\SmartSearchReady";
    public const int GiveUpSeconds = 180;   // the splash never stays forever

    [STAThread]
    static int Main(string[] args)
    {
        string root = AppDomain.CurrentDomain.BaseDirectory;
        string py = null;
        foreach (var c in new[] { @"python\pythonw.exe", @".venv\Scripts\pythonw.exe" })
            if (File.Exists(Path.Combine(root, c))) { py = Path.Combine(root, c); break; }

        if (py == null || !File.Exists(Path.Combine(root, "main.py")) || !Directory.Exists(Path.Combine(root, "app")))
        {
            MessageBox.Show("Не найдены файлы программы (папки python и app рядом с SmartSearch.exe).\n\n" +
                            "Распакуйте архив целиком в обычную папку и запускайте SmartSearch.exe из неё.",
                            Title, MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }

        var psi = new ProcessStartInfo(py)
        {
            Arguments = "main.py --desktop" + (args.Length > 0 ? " " + string.Join(" ", args) : ""),
            WorkingDirectory = root,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        psi.EnvironmentVariables["PYTHONUTF8"] = "1";
        psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
        psi.EnvironmentVariables["HF_HUB_OFFLINE"] = "1";

        EventWaitHandle ready = null;
        try { ready = new EventWaitHandle(false, EventResetMode.ManualReset, ReadyEvent); ready.Reset(); }
        catch (Exception) { ready = null; }   // no splash wait then, just start

        Process proc;
        try { proc = Process.Start(psi); }
        catch (Exception e)
        {
            MessageBox.Show("Не удалось запустить программу:\n\n" + e.Message, Title,
                            MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 2;
        }
        if (ready == null || proc == null) return 0;

        Application.EnableVisualStyles();
        Application.Run(new Splash(root, proc, ready));
        return 0;
    }
}

class Splash : Form
{
    readonly Process proc;
    readonly EventWaitHandle ready;
    readonly string root;
    readonly Label status;
    readonly Label timeLabel;
    readonly ProgressBar bar;
    readonly System.Windows.Forms.Timer timer;
    readonly DateTime started = DateTime.Now;
    bool failed;

    static readonly Color Accent = Color.FromArgb(79, 70, 229);
    static readonly Color Muted = Color.FromArgb(110, 110, 125);

    public Splash(string root, Process proc, EventWaitHandle ready)
    {
        this.root = root; this.proc = proc; this.ready = ready;
        Text = "Bilingual Text Analyzer — запуск";
        FormBorderStyle = FormBorderStyle.None;
        StartPosition = FormStartPosition.CenterScreen;
        ShowInTaskbar = true;
        TopMost = true;
        BackColor = Color.White;
        Size = new Size(440, 230);
        try { Icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath); } catch (Exception) { }

        var pic = new PictureBox { Location = new Point(28, 30), Size = new Size(64, 64), SizeMode = PictureBoxSizeMode.Zoom };
        string png = Path.Combine(root, @"app\static\icon.png");
        try { if (File.Exists(png)) pic.Image = Image.FromFile(png); else if (Icon != null) pic.Image = Icon.ToBitmap(); } catch (Exception) { }
        Controls.Add(pic);

        Controls.Add(new Label { Text = "Bilingual Text Analyzer", Location = new Point(108, 34), AutoSize = true,
                                 Font = new Font("Segoe UI Semibold", 15f), ForeColor = Color.FromArgb(30, 30, 40) });
        Controls.Add(new Label { Text = "SmartSearch — умный поиск по книгам", Location = new Point(110, 68), AutoSize = true,
                                 Font = new Font("Segoe UI", 9.5f), ForeColor = Muted });

        status = new Label { Text = "Программа запускается…", Location = new Point(28, 120), Size = new Size(300, 22),
                             Font = new Font("Segoe UI", 10f), ForeColor = Color.FromArgb(30, 30, 40) };
        timeLabel = new Label { Text = "", Location = new Point(330, 120), Size = new Size(82, 22), TextAlign = ContentAlignment.TopRight,
                                Font = new Font("Segoe UI", 9.5f), ForeColor = Muted };
        bar = new ProgressBar { Style = ProgressBarStyle.Marquee, MarqueeAnimationSpeed = 25,
                                Location = new Point(28, 148), Size = new Size(384, 8) };
        Controls.Add(status); Controls.Add(timeLabel); Controls.Add(bar);
        Controls.Add(new Label { Text = "Первый запуск после загрузки компьютера может занять до полуминуты.",
                                 Location = new Point(28, 172), Size = new Size(384, 36), Font = new Font("Segoe UI", 8.5f), ForeColor = Muted });

        // drag the borderless window by any place
        MouseDown += Drag;
        foreach (Control c in Controls) if (!(c is ProgressBar)) c.MouseDown += Drag;

        timer = new System.Windows.Forms.Timer { Interval = 200 };
        timer.Tick += Tick;
        timer.Start();
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e);
        using (var pen = new Pen(Accent, 2)) e.Graphics.DrawRectangle(pen, 1, 1, Width - 3, Height - 3);
        using (var b = new SolidBrush(Accent)) e.Graphics.FillRectangle(b, 0, 0, Width, 5);
    }

    void Tick(object sender, EventArgs e)
    {
        if (failed) return;
        int s = (int)(DateTime.Now - started).TotalSeconds;
        timeLabel.Text = s > 0 ? s + " с" : "";
        if (s >= 4) status.Text = "Загружаются модули поиска…";
        if (s >= 12) status.Text = "Почти готово, открывается окно…";

        if (ready.WaitOne(0)) { timer.Stop(); Close(); return; }
        bool exited;
        try { exited = proc.HasExited; } catch (Exception) { exited = false; }
        if (exited)
        {
            // a restart from the settings page or a second copy may still raise the event a moment later
            failed = true; timer.Stop();
            bar.Style = ProgressBarStyle.Continuous; bar.Value = 0;
            status.ForeColor = Color.FromArgb(190, 30, 30);
            status.Text = "Программа не запустилась.";
            ShowError("Подробности — в файлах папки logs:\n" + Path.Combine(root, "logs"));
            return;
        }
        if (s >= Launcher.GiveUpSeconds) { timer.Stop(); Close(); }
    }

    void ShowError(string details)
    {
        TopMost = false;
        MessageBox.Show(this, "Программа не смогла запуститься.\n\n" + details, "Bilingual Text Analyzer",
                        MessageBoxButtons.OK, MessageBoxIcon.Error);
        Close();
    }

    [System.Runtime.InteropServices.DllImport("user32.dll")] static extern bool ReleaseCapture();
    [System.Runtime.InteropServices.DllImport("user32.dll")] static extern IntPtr SendMessage(IntPtr h, int msg, int wp, int lp);
    void Drag(object sender, MouseEventArgs e)
    {
        if (e.Button != MouseButtons.Left) return;
        ReleaseCapture();
        SendMessage(Handle, 0xA1, 2, 0);   // WM_NCLBUTTONDOWN, HTCAPTION
    }
}
