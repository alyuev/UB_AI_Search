// SmartSearch.exe: starts the program in its own window, without a console.
// Built by scripts/build_launcher.py with the C# compiler that ships with Windows (.NET Framework 4.x).
using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

[assembly: System.Reflection.AssemblyTitle("Bilingual Text Analyzer")]
[assembly: System.Reflection.AssemblyProduct("Bilingual Text Analyzer")]

static class Launcher
{
    const string Title = "Bilingual Text Analyzer";

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
        try
        {
            Process.Start(psi);
            return 0;
        }
        catch (Exception e)
        {
            MessageBox.Show("Не удалось запустить программу:\n\n" + e.Message, Title,
                            MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 2;
        }
    }
}
