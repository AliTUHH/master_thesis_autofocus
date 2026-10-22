# GPU-Läufe auf DESY Maxwell – Anleitung

Diese Anleitung beschreibt, wie du die rechenintensiven Läufe der Arbeit auf den A100-Knoten
von Maxwell startest und ihre Ergebnisse prüfst: die Baseline von [2] auf dem ganzen
synthetischen Test-Split, größere Trainingsläufe und NPE (Etappe 7). Die Werkzeuge liegen in
`cluster/maxwell/`: SLURM-Jobs, ein Skript für die Python-Umgebung und
`scripts/check_cluster_run.py`, das Läufe auf Maxwell Sample für Sample mit den committeten
Ergebnissen der VM vergleicht.

Die VM, auf der der Code entsteht, erreicht DESY nicht. Alles, was von Maxwell selbst abhängt
(Partitionen, Knoten-Features, Module, Treiber, Pfade, Kontingente, Zeitlimits), habe ich
nicht geraten, sondern als **PRÜFEN** markiert (Abschnitt 2). In der Einstellungsdatei steht
dort `CHECK_ME`, und die Skripte starten nicht, solange ein `CHECK_ME` übrig ist. Hast du für
deinen PyTorch-Test auf der A100 schon Partition, Constraint und Module gefunden, nimm
dieselben. Was auf der VM geprüft werden konnte und was nicht, steht in Abschnitt 11.

## 1 Überblick

Reihenfolge:

1. Einstellungen auf Maxwell nachsehen und in `cluster/maxwell/config.sh` eintragen
   (Abschnitte 2 und 3).
2. Python-Umgebung einrichten (Abschnitt 3).
3. Einen GPU-Knoten prüfen (Abschnitt 4).
4. Val- und Test-Split auf der CPU erzeugen, Trainingsdaten auf der GPU (Abschnitt 5).
5. Baseline auf dem ganzen Test-Split (Abschnitt 6), Training (Abschnitt 7), NPE und weitere
   Läufe (Abschnitt 8).
6. Ergebnisse auf einem Branch zurückgeben (Abschnitt 9).

| Was | Wo | Warum |
|---|---|---|
| Klon des Repos (Code, Configs, committete Ergebnisse) | beliebig, z. B. im Home-Verzeichnis | klein (etwa 40 MB) |
| Python-Umgebung | Scratch: `$HOLOFOCUS_SCRATCH/env` | einige GB mit den CUDA-Wheels, jederzeit neu erzeugbar |
| Download-Caches von pip und conda | Scratch: `$HOLOFOCUS_SCRATCH/cache` | sonst landen die CUDA-Wheels im Home-Verzeichnis; darf gelöscht werden |
| Datensätze (HDF5) | Scratch: `$HOLOFOCUS_SCRATCH/data/<name>/` | 0,7 MB pro Sample, zusammen etwa 41 GB (Abschnitt 5), aus den Configs reproduzierbar |
| Laufverzeichnisse, Logs | Scratch: `$HOLOFOCUS_SCRATCH/runs/`, `.../logs/` | Arbeitsbereich der Jobs |
| Ergebnisse, die nicht verloren gehen dürfen | DUST: `$HOLOFOCUS_DUST/holofocus/runs/...` | jeder Job außer `run` kopiert am Ende sein Laufverzeichnis ohne HDF5-Dateien dorthin (wenige MB) |
| Ergebnisse für die Arbeit | Git: `results/maxwell/` auf einem eigenen Branch | klein; von dort kommen sie ins Repo und ins Laborbuch (Abschnitt 9) |

Die Jobs liegen in `cluster/maxwell/jobs/` und werden mit
`cluster/maxwell/submit.sh [Optionen] JOB [Argumente]` gestartet:

| Job | Zweck | Zeitlimit |
|---|---|---|
| `gpu-check` | prüft einen GPU-Knoten gegen die CPU und gegen die committeten Ergebnisse (Abschnitt 4) | 1 h |
| `datagen` | erzeugt die Splits einer Datensatz-Config (Abschnitt 5) | 4 h |
| `predict` | Schätzungen eines gelernten Modells für einen Split, mit Vergleich gegen die committeten | 1 h |
| `baseline` | klassischer Autofokus von [2] auf einem Split, als Job-Array in `--chunks N` Stücken (Abschnitt 6) | 24 h |
| `merge` | fügt die Stücke eines Baseline-Arrays zu einem Laufverzeichnis zusammen | 15 min |
| `train` | trainiert ein Modell und sagt Val und Test vorher, als Array mit einem Seed pro Task (Abschnitt 7) | 12 h |
| `npe` | trainiert einen NPE (Etappe 7) und sagt Val, Test und den Kalibrier-Split vorher, mit den Prüfungen von sbi; als Array ein Seed pro Task (Abschnitt 8.1) | 12 h |
| `run` | beliebiger Befehl in der Umgebung, z. B. ein MFE-Scan (Abschnitt 8) | 8 h |

Die Zeitlimits stehen als `#SBATCH --time` in den Job-Skripten; `--time` ersetzt sie.
Optionen von `submit.sh`, vor dem Namen des Jobs:

| Option | Wirkung |
|---|---|
| `--cpu` | CPU-Partition ohne GPU, die Jobs rechnen mit `--device cpu` |
| `--time T` | Zeitlimit, z. B. `--time 12:00:00` |
| `--array SPEC` | Job-Array, z. B. `0-9` oder `3`; bei `baseline` rechnet Task k Stück k von `--chunks N`, bei `train` und `npe` trainiert Task k mit `--seed k` |
| `--after ID` | startet erst, wenn Job `ID` (bei einem Array alle Tasks) erfolgreich beendet ist |
| `--sbatch OPT` | weitere sbatch-Option, z. B. `--sbatch --mem=32G` (wiederholbar) |
| `--dry-run` | zeigt den sbatch-Befehl, statt ihn abzuschicken |
| `--local` | führt den Job ohne SLURM in der aktuellen Shell aus, mit `--device cpu` (für kurze Prüfungen auf dem Login-Knoten) |

Pfade in den Argumenten: Configs relativ zum Repo (`configs/...`), Datendateien relativ zu
`$HOLOFOCUS_SCRATCH/data` (`cpu-4x/test.h5`), Laufverzeichnisse relativ zu
`$HOLOFOCUS_SCRATCH/runs` (`baseline/cpu-4x-test-p2`); absolute Pfade gehen immer.
`submit.sh --help` und der Kopf jedes Job-Skripts nennen alle Argumente.

`submit.sh` gibt die Job-ID allein auf stdout aus und den Namen der Log-Datei auf stderr. So
lassen sich Jobs verketten:

```bash
id=$(cluster/maxwell/submit.sh gpu-check)   # stderr: submitted gpu-check as job 1234, log ...
cluster/maxwell/submit.sh --cpu --after "$id" ...   # startet erst nach Job 1234
```

Die Logs heißen `$HOLOFOCUS_SCRATCH/logs/holofocus-<job>-<job-id>.out`, bei Job-Arrays
`holofocus-<job>-<array-id>_<task>.out`. Jedes Log beginnt mit Datum, Knoten, Partition, Zahl
der CPUs, Commit des Repos, Gerät, GPU und Versionen.

## 2 Was du auf Maxwell prüfen musst (PRÜFEN)

Alle Einstellungen kommen in `cluster/maxwell/config.sh` (Abschnitt 3). Die Befehle in der
Tabelle sind Standardbefehle von SLURM und Linux; die DESY-Dokumentation zu Maxwell und der
Support sind die zweite Quelle.

| Einstellung | Was | So prüfst du es |
|---|---|---|
| `HOLOFOCUS_GPU_PARTITION` | Partition mit A100-Knoten, die dein Konto nutzen darf | `sinfo -s` listet die Partitionen; `sinfo -o "%20P %10l %6D %40f %20G"` zeigt pro Partition Zeitlimit, Knotenzahl, Features und GRES |
| `HOLOFOCUS_GPU_CONSTRAINT` | Feature-Name der A100-Knoten für `--constraint` | Spalte Features oben oder `scontrol show node <knoten> \| grep -i features`; leer (`""`), wenn die Partition nur A100-Knoten hat |
| `HOLOFOCUS_GPU_REQUEST` | sbatch-Option für eine GPU | `scontrol show node <knoten> \| grep -i gres`: Steht dort `Gres=gpu:...`, ist die übliche SLURM-Form `--gres=gpu:1`. Steht dort `Gres=(null)`, kommt ein Knoten mit seinen GPUs, und der Wert bleibt leer |
| GPUs pro Knoten, ganze Knoten | ob ein Job einen ganzen Knoten bekommt | `sinfo -p <partition> -N -o "%20N %6c %10m %30G"` (CPUs, Speicher in MB und GRES pro Knoten); `scontrol show partition <partition> \| grep -i oversubscribe` (`EXCLUSIVE` heißt ganze Knoten). Jeder Job nutzt eine GPU; bekommt er einen Knoten mit mehreren, bleiben die übrigen ungenutzt |
| CPUs und Arbeitsspeicher pro Job | die Jobs verlangen weder `--cpus-per-task` noch `--mem`, sie bekommen, was die Partition vorgibt | `scontrol show partition <partition>`, Felder `DefMemPerCPU`, `DefMemPerNode` und `MaxMemPerNode`; im Log zeigt die Zeile `host` die CPUs und `OMP_NUM_THREADS`. Bekommt ein Job nur eine CPU oder wenig Speicher, z. B. `--cpus-per-task=8 --mem=32G` in `HOLOFOCUS_SBATCH_EXTRA` eintragen oder einzeln mit `--sbatch` angeben |
| `HOLOFOCUS_CPU_PARTITION` | Partition für Jobs ohne GPU (`--cpu`) | `sinfo -s`; darf dieselbe sein wie die GPU-Partition |
| Zeitlimit | wie lange ein Job laufen darf | `sinfo -o "%20P %10l"`; die Jobs verlangen die Zeiten aus Abschnitt 1. Ist das Limit kleiner, `--time` angeben und die Baseline in mehr Stücke teilen (Abschnitt 6) |
| `HOLOFOCUS_SBATCH_EXTRA` | weitere Optionen für jeden Job, z. B. ein Konto | `sacctmgr show assoc user=$USER format=account%20,partition%20`; dann `"--account=<konto>"` |
| `HOLOFOCUS_MODULES` | Module für Python 3.12 oder conda | `module avail`, `module avail python`, `module avail conda`. Python muss mindestens 3.12 sein, weil numpy 2.5.4 und scipy 1.18.1 der VM es brauchen. Ein CUDA-Modul ist nicht nötig: Die PyTorch-Wheels bringen die CUDA-Bibliotheken mit, gebraucht wird nur der NVIDIA-Treiber des Knotens. Leer, wenn `python3 --version` schon passt oder du ein eigenes Miniforge nimmst |
| `HOLOFOCUS_TORCH_INDEX` | CUDA-Version der PyTorch-Wheels | auf einem GPU-Knoten `nvidia-smi` (z. B. `srun -p <partition> [--constraint=...] [--gres=...] --time=00:10:00 nvidia-smi`): Oben rechts steht „CUDA Version: X.Y“, die höchste CUDA-Version des Treibers. Für torch 2.14.1 gibt es `cu126`, `cu130` und `cu132`; nimm die höchste, die nicht über X.Y liegt. Bei 12.0 bis 12.5 läuft `cu126` meist trotzdem (Minor-Version-Kompatibilität von CUDA 12), der gpu-check zeigt es; sonst Abschnitt 10 |
| glibc | die Wheels von torch 2.14.1 brauchen glibc ≥ 2.28 | `getconf GNU_LIBC_VERSION` (gibt auch `setup_env.sh` aus) |
| Internet auf dem Login-Knoten | pip lädt PyTorch und die Pakete | `curl -sI https://download.pytorch.org/whl/ \| head -n 1` sollte `HTTP/2 200` zeigen; sonst beim Support nach einem Proxy fragen |
| `HOLOFOCUS_SCRATCH` | dein Ordner unter `/gpfs/petra3/scratch/` | `ls -ld /gpfs/petra3/scratch/<ordner>`, freier Platz mit `df -h /gpfs/petra3/scratch/<ordner>`; alle Daten aus Abschnitt 5 mit Umgebung und Caches brauchen etwa 50 GB. Auch prüfen: Werden alte Dateien dort automatisch gelöscht? Alles außer den Ergebnissen ist reproduzierbar |
| `HOLOFOCUS_DUST` | dein DUST-Verzeichnis | Pfad und Kontingent (`df -h <pfad>`); die Ergebnisse sind klein (ein Baseline-Lauf auf 250 Samples mit seinen Stücken etwa 1 MB, ein Trainingslauf 2 MB) |

## 3 Einmalige Einrichtung

```bash
git clone https://github.com/AliTUHH/master_thesis_autofocus.git
cd master_thesis_autofocus
cp cluster/maxwell/config.example.sh cluster/maxwell/config.sh
nano cluster/maxwell/config.sh     # alle CHECK_ME ersetzen (Abschnitt 2)
cluster/maxwell/setup_env.sh       # auf dem Login-Knoten, braucht Internet
```

`setup_env.sh` lädt die Module, legt die Umgebung an (Standard: venv in
`$HOLOFOCUS_SCRATCH/env`) und installiert PyTorch (`HOLOFOCUS_TORCH_SPEC` aus
`HOLOFOCUS_TORCH_INDEX`), die Paketversionen der VM (`cluster/maxwell/requirements-vm.txt`;
numpy bestimmt die Zufallsfolgen und damit die Labels) und das Repo als editierbares Paket.
Die Downloads, mit den CUDA-Wheels mehrere GB, legen pip und conda in
`$HOLOFOCUS_SCRATCH/cache` ab statt im Home-Verzeichnis. Auf der VM dauerte die Einrichtung
mit den CPU-Wheels 34 s; mit den CUDA-Wheels kommt vor allem die Download-Zeit dazu. Am Ende
stehen die Versionen in `$HOLOFOCUS_ENV/holofocus-env.txt`. Erwartet (mit deiner
CUDA-Variante):

```
torch 2.14.1+cu126 CUDA build 12.6 cuDNN ...
CUDA available here False (no GPU on a login node is normal)
numpy 2.5.4 holofocus 0.1.0
```

Varianten:

- **conda statt venv:** `HOLOFOCUS_ENV_KIND="conda"` und `HOLOFOCUS_CONDA` auf das
  conda-Programm setzen (aus einem Modul oder einem eigenen Miniforge, z. B.
  `curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh`
  und `bash Miniforge3-Linux-x86_64.sh -b -p $HOLOFOCUS_SCRATCH/miniforge3`, dann
  `HOLOFOCUS_CONDA="$HOLOFOCUS_SCRATCH/miniforge3/bin/conda"`). Die Umgebung bekommt Python 3.12
  wie die VM. Das ist auch der Weg, wenn kein Modul Python 3.12 bietet.
- **eigene, schon funktionierende Umgebung:** `HOLOFOCUS_ENV_KIND="none"`; die Jobs nehmen dann
  das `python` aus dem `PATH` nach `module load $HOLOFOCUS_MODULES`. In dieser Umgebung einmal
  `pip install -c cluster/maxwell/requirements-vm.txt -e ".[dev]"` ausführen.
- **neu aufsetzen:** `setup_env.sh --recreate`.
- **sbi für NPE (Etappe 7):** `setup_env.sh --with-sbi` installiert das Extra `npe` mit den
  Versionen der VM (Abschnitt 8.1); torch und die gepinnten Pakete bleiben dabei, wie sie sind.

In jeder neuen Shell auf Maxwell hilft `source cluster/maxwell/config.sh`: Dann stehen
`$HOLOFOCUS_SCRATCH`, `$HOLOFOCUS_DUST` und `$HOLOFOCUS_ENV` auch in deinen eigenen Befehlen
zur Verfügung. Die Befehle der folgenden Abschnitte setzen das voraus.

## 4 GPU-Test

```bash
cluster/maxwell/submit.sh --dry-run gpu-check   # zeigt den sbatch-Befehl, startet nichts
cluster/maxwell/submit.sh gpu-check             # gibt die Job-ID aus
squeue -u $USER                                 # PD = wartet, R = läuft
less $HOLOFOCUS_SCRATCH/logs/holofocus-gpu-check-<job-id>.out
```

Im `--dry-run` siehst du Partition, Constraint und GPU-Anforderung so, wie sbatch sie
bekommt. Wie lange der Job auf einer A100 braucht, ist nicht gemessen; auf der CPU der VM
waren es 5,5 min mit 4 Threads. Das Log beginnt mit Knoten, GPU, Treiber und Versionen; danach folgen die
Prüfungen, jede mit PASS, WARN oder FAIL:

| Prüfung | erwartet | Bedeutung |
|---|---|---|
| `device` | PASS, A100 mit 40 oder 80 GB, compute capability 8.0 | PyTorch sieht die GPU |
| `fft2 1024^2` | PASS (Abweichung < 1e-5) | cuFFT rechnet wie die CPU |
| `conv1d` | PASS | ohne TF32 (der Kit setzt `NVIDIA_TF32_OVERRIDE=0`); WARN mit etwa 1e-3 heißt, TF32 ist an |
| `forward model` | PASS (< 1e-4) | die Simulation auf der GPU stimmt mit der CPU; das verrauschte Hologramm darf abweichen |
| `ASRM padding 2` / `4` | PASS, mit der Zeit pro Rekonstruktion | die Rekonstruktion von [2] auf der GPU; bei Padding 2 der MFE auf 1e-3 wie auf der CPU |
| `data labels` | PASS | die ersten vier Test-Samples, auf der GPU erzeugt, haben dieselben Labels wie auf der VM |
| `data holograms` | PASS oder WARN | auf der GPU simuliert können einzelne Rauschwerte abweichen |
| `predictions` | PASS (≤ 1 µm) oder WARN | das committete Hauptmodell schätzt auf der GPU wie auf der VM |
| `training` | PASS | das Training läuft auf der GPU (3 Epochen der CI-Config auf den vier Samples); über die Güte sagt das nichts |
| `baseline` | PASS (≤ 1 µm) oder WARN | der Autofokus von Sample 0 nimmt denselben Weg wie in Etappe 3. Bei einem einzelnen Sample gibt es kein FAIL, weil ein anderer Weg durch Rundung (6.4) nicht von einer anderen Rechnung zu unterscheiden ist. Dazu die Zeit pro Rekonstruktion gegenüber der VM (dort 31 s mit 3 Threads) |

Zum Schluss steht „no check failed“. Für die Planung brauchst du zwei Zeilen:

- `padding_2: baseline on 250 test samples about … h on this device`, ebenso für
  `padding_4`: die Dauer der Baseline auf dem ganzen Test-Split (12,1 Rekonstruktionen pro
  Sample wie in Etappe 3). Danach richtest du die Zahl der Stücke in Abschnitt 6 aus.
- `test: 4/4 samples, … s/sample`: die Zeit pro Sample beim Erzeugen auf der GPU, mit dem
  Start von CUDA, also eher zu hoch (Abschnitt 5).

Bei FAIL siehe Abschnitt 10. Nach jeder Änderung an der Umgebung (`setup_env.sh --recreate`,
andere Module, anderer Index) lohnt es sich, den gpu-check zu wiederholen.

## 5 Daten erzeugen

| Config | Split | Samples | Größe | erzeugen auf |
|---|---|---|---|---|
| `configs/dataset/cpu-4x.yaml` | `val`, `test` | je 250 | je 183 MB | CPU (`--cpu`), damit sie den Splits der committeten Ergebnisse entsprechen |
| `configs/dataset/cpu-4x.yaml` | `train` | 2000 | 1,5 GB | GPU |
| `configs/dataset/cpu-4x-extra.yaml` | `train` | 8000 | 5,4 GB | GPU |
| `configs/dataset/cpu-4x-large.yaml` | `train` | 50 000 | etwa 34 GB | GPU |

```bash
cluster/maxwell/submit.sh --cpu datagen configs/dataset/cpu-4x.yaml --splits val test
cluster/maxwell/submit.sh datagen configs/dataset/cpu-4x.yaml --splits train
cluster/maxwell/submit.sh datagen configs/dataset/cpu-4x-extra.yaml
cluster/maxwell/submit.sh --time 12:00:00 datagen configs/dataset/cpu-4x-large.yaml
```

Der erste Job vergleicht Val und Test mit ihren Fingerabdrücken in `results/datasets/`
(Labels, dazu SHA-256, Mittelwert und Standardabweichung jedes Hologramms der VM). Erwartet:
`data labels` PASS und `data holograms` PASS (bitgleich) oder WARN (anderer CPU-Typ, einzelne
Rauschwerte). FAIL bei den Labels heißt: andere numpy-Version oder geänderte Config; dann den
Split nicht mit den committeten Ergebnissen vergleichen. Auf der VM dauerte ein Split etwa
1,5 min.

Wichtig: `cpu-4x.yaml` auf der GPU immer mit `--splits train` starten. Ohne `--splits`
entstehen auch Val und Test auf der GPU, und ein späterer `--cpu`-Lauf überspringt sie, weil
die Config dieselbe ist (`up to date, skipped`). Dann die beiden Dateien löschen und mit
`--cpu` neu erzeugen.

Trainingsdaten dürfen auf der GPU entstehen: Die Labels sind dieselben, einzelne Rauschwerte
können abweichen, was für das Training keine Rolle spielt. `cpu-4x-extra` und `cpu-4x-large`
haben eigene Seeds, ihre Samples sind also unabhängig von denen in `cpu-4x`, und speichern
keine Phasenkarten. Ein Split entsteht in einem Stück: Bis zum Ende heißt die Datei
`train.h5.part`, und ein abgebrochener Split beginnt von vorn. Rechne deshalb vorher nach:
50 000 mal die Zeit pro Sample aus dem gpu-check muss deutlich unter dem Zeitlimit bleiben.
Auf der CPU der VM waren es 0,37–0,55 s pro Sample, für 50 000 Samples also 5–8 h; die
12 h im Befehl oben lassen dafür Platz. Vorhandene Splits mit derselben Config überspringt
der Job.

## 6 Baseline auf dem ganzen Test-Split

Das ist der wichtigste GPU-Lauf (Laborbuch 04, 9.6): der Autofokus von [2] auf allen 250
Test-Samples statt auf 48, kalt und mit Warmstart, dazu der Offset der korrigierten Baseline
auf allen 250 Val-Samples statt auf 24. Erst damit ist der Anteil innerhalb 100 µm für
Baseline und Warmstart auf etwa ±6 statt ±13 Prozentpunkte bestimmt, so genau wie für das
Netz. Die Läufe mit Padding 4 statt 2 (6.2) klären den offenen Punkt 8.6 aus Laborbuch 04:
Bei zwei von drei geprüften Zellen nahm Padding 4 den größten Teil der Verschiebung des
MFE-Minimums weg.

Jedes Array läuft in `--chunks 10` Stücken zu 25 Samples (`--array 0-9`); Task k rechnet
Stück k nach `<name>/chunk-k-of-10`. `--array` und `--chunks` müssen zusammenpassen
(`--array 0-(N-1)` mit `--chunks N`). Ein Stück sollte höchstens die Hälfte des Zeitlimits
brauchen: Der gpu-check schätzt die Dauer für 250 Samples, ein Stück braucht ein Zehntel davon
(mit Warmstart etwa drei Viertel davon). Bei einem kurzen Zeitlimit nimmst du mehr Stücke,
z. B. `--array 0-24` mit `--chunks 25`.

### 6.1 Pflicht: Padding 2 wie auf der VM

```bash
source cluster/maxwell/config.sh
R=$HOLOFOCUS_SCRATCH/runs
# committete Schätzungen des Hauptmodells für den Test-Split: derselbe Warmstart wie auf der VM
W=results/learned/spectrum-cnn-wide/test/predictions.csv

# kalter Start wie in [2]
id=$(cluster/maxwell/submit.sh --array 0-9 baseline baseline/cpu-4x-test-p2 \
    configs/baseline/paper2-cpu.yaml cpu-4x/test.h5 --chunks 10)
cluster/maxwell/submit.sh --cpu --after "$id" merge baseline/cpu-4x-test-p2

# Warmstart: Nelder–Mead beginnt bei z01_est ± 0,5 mm statt an den Rändern von Omega_z
id=$(cluster/maxwell/submit.sh --array 0-9 baseline baseline/cpu-4x-test-warm-p2 \
    configs/baseline/paper2-cpu-warm.yaml cpu-4x/test.h5 --chunks 10 \
    --warm-start "$W" --half-width 5e-4)
cluster/maxwell/submit.sh --cpu --after "$id" merge baseline/cpu-4x-test-warm-p2

# Offset der korrigierten Baseline: alle 250 Val-Samples
id=$(cluster/maxwell/submit.sh --array 0-9 baseline baseline/cpu-4x-val-p2 \
    configs/baseline/paper2-cpu.yaml cpu-4x/val.h5 --chunks 10)
cluster/maxwell/submit.sh --cpu --after "$id" merge baseline/cpu-4x-val-p2
```

`$W` gehört zum Test-Split der VM. Hatten die Labels beim Erzeugen auf Maxwell PASS
(Abschnitt 5), passt die Datei auch zum Test-Split auf Maxwell. Dann beginnt der Warmstart bei
jedem Sample mit demselben Simplex wie auf der VM, und der Lauf lässt sich Sample für Sample
mit `results/baseline/cpu-4x-test-warm` vergleichen (6.4).

### 6.2 Empfohlen: Padding 4 (Laborbuch 04, Abschnitt 7 und 8.6)

```bash
for split in test val; do
    id=$(cluster/maxwell/submit.sh --array 0-9 baseline baseline/cpu-4x-$split-p4 \
        configs/baseline/paper2-cpu-p4.yaml cpu-4x/$split.h5 --chunks 10)
    cluster/maxwell/submit.sh --cpu --after "$id" merge baseline/cpu-4x-$split-p4
done
id=$(cluster/maxwell/submit.sh --array 0-9 baseline baseline/cpu-4x-test-warm-p4 \
    configs/baseline/paper2-cpu-warm-p4.yaml cpu-4x/test.h5 --chunks 10 \
    --warm-start "$W" --half-width 5e-4)
cluster/maxwell/submit.sh --cpu --after "$id" merge baseline/cpu-4x-test-warm-p4
```

Die beiden Configs sind `paper2-cpu.yaml` und `paper2-cpu-warm.yaml` mit Padding 4 (Arrays
von 2048² statt 1024²); ein Test stellt sicher, dass sie sich sonst nicht unterscheiden. Auf
der CPU dauert eine Rekonstruktion damit etwa 190 statt 29 s, deshalb gibt es auf der VM
keinen Baseline-Lauf mit Padding 4 zum Vergleichen. Zur Orientierung: Das MFE-Minimum der
Test-Samples 0, 1 und 2 lag mit Padding 4 bei −33, +2 und +121 µm (Laborbuch 04,
Abschnitt 7); wie genau Nelder–Mead ein so flaches Minimum trifft, ist offen.

### 6.3 Optional: die Schätzung als Ecke des Simplex (Laborbuch 04, 9.1)

```bash
id=$(cluster/maxwell/submit.sh --array 0-9 baseline baseline/cpu-4x-test-warm-vertex-p2 \
    configs/baseline/paper2-cpu-warm.yaml cpu-4x/test.h5 --chunks 10 \
    --warm-start "$W" --half-width 5e-4 --anchor vertex)
cluster/maxwell/submit.sh --cpu --after "$id" merge baseline/cpu-4x-test-warm-vertex-p2
```

Mit `--anchor vertex` beginnt Nelder–Mead bei (z01_est, z01_est + 0,5 mm), sodass die
Schätzung selbst die erste Probe ist. Auf der VM gibt es diesen Lauf nur für die Samples 30,
35, 40 und 41.

### 6.4 Zusammenführen, wiederholen, prüfen

`merge` startet erst, wenn alle Tasks des Arrays erfolgreich beendet sind, und schreibt dann
`results.csv`, `traces.jsonl`, `summary.json` und `run.yaml` des ganzen Laufs nach
`$R/baseline/<name>/`, nach Sample-Index sortiert; die `chunk-*`-Verzeichnisse bleiben
daneben. Das Log nennt pro Stück die Zahl der Samples, Gerät und Knoten und am Ende z. B.
`250 samples (indices 0 .. 249) from 10 runs`.

Bricht ein Stück ab (Zeitlimit, Knotenfehler), bleibt `merge` in `squeue` mit dem Grund
`DependencyNeverSatisfied` stehen. Dann:

```bash
scancel <id-des-merge-jobs>
# Stück 3 noch einmal: derselbe Befehl mit --array 3, er setzt beim nächsten fehlenden Sample fort
id=$(cluster/maxwell/submit.sh --array 3 baseline baseline/cpu-4x-test-p2 \
    configs/baseline/paper2-cpu.yaml cpu-4x/test.h5 --chunks 10)
cluster/maxwell/submit.sh --cpu --after "$id" merge baseline/cpu-4x-test-p2
```

`merge` lässt sich jederzeit wiederholen, auch direkt auf dem Login-Knoten
(`cluster/maxwell/submit.sh --local merge baseline/cpu-4x-test-p2`, Sekunden). Ein
Laufverzeichnis, das schon einen Lauf mit anderer Config, anderen Daten oder anderem Warmstart
enthält, lehnt `baseline` ab; ebenso lehnt `merge` Stücke verschiedener Läufe ab
(Abschnitt 10).

Prüfen, Sample für Sample gegen die Läufe der VM (auf dem Login-Knoten, wenige Sekunden):

```bash
check() {  # check LAUF REFERENZ
    cluster/maxwell/submit.sh --local run python scripts/check_cluster_run.py baseline \
        "$R/baseline/$1" --reference "results/baseline/$2"
}
check cpu-4x-test-p2 cpu-4x-test                       # 48 Samples, kalter Start (Etappe 3)
check cpu-4x-test-warm-p2 cpu-4x-test-warm             # 48 Samples, Warmstart (Etappe 4)
check cpu-4x-val-p2 cpu-4x-val                         # 24 Samples, Offset (Etappe 4)
check cpu-4x-test-warm-vertex-p2 cpu-4x-test-warm-vertex   # 4 Samples, nur nach 6.3
```

Verglichen werden die Samples, die beide Läufe enthalten. Erwartet ist PASS oder WARN:
Nelder–Mead nimmt bei einzelnen Samples einen anderen Weg, wenn zwei MFE-Werte durch Rundung
die Reihenfolge tauschen. Die Schätzung weicht dann um bis zu eine Simplexlänge (100 µm) ab,
oder um mehr, wenn der andere Weg in einer anderen Senke des MFE endet (Laborbuch 04, 5.8).
FAIL heißt, dass der Median der Abweichungen über mindestens 3 Samples über 1 µm liegt, also
etwas anders gerechnet wird. Die Zeile `time per reconstruction … here, … in the reference`
zeigt die Beschleunigung gegenüber der VM.

Wer den Login-Knoten nicht belasten will, startet dieselben Befehle ohne `--local` und mit
`--cpu` als Job. Optional prüft der Job `predict` das Hauptmodell auf dem Test-Split von
Maxwell (PASS bei höchstens 1 µm Abweichung von den committeten Schätzungen); für den
Warmstart wird seine Ausgabe nicht gebraucht:

```bash
cluster/maxwell/submit.sh predict results/learned/spectrum-cnn-wide cpu-4x/test.h5 \
    learned/spectrum-cnn-wide-test
```

### 6.5 Vergleich mit dem Netz

```bash
for p in p2 p4; do
    cluster/maxwell/submit.sh --local run holofocus-learn compare --learned "$W" \
        --baseline "$R/baseline/cpu-4x-test-$p" --baseline-val "$R/baseline/cpu-4x-val-$p" \
        --warm "$R/baseline/cpu-4x-test-warm-$p" --out "$R/comparison/cpu-4x-test-$p"
done
```

`comparison.json` enthält pro Verfahren (Netz, rohe und korrigierte Baseline, Warmstart roh
und korrigiert) die Fehlerkennzahlen mit den Anteilen innerhalb 100/250/1000 µm,
Rekonstruktionen und Zeit pro Schätzung, dazu den Offset mit Standardfehler und
Bootstrap-Intervall und die gepaarten Differenzen; `comparison.csv` hat eine Zeile pro Sample.
Auf der VM ergibt derselbe Befehl mit den committeten Läufen genau
`results/comparison/cpu-4x-test/`.

## 7 Größere Trainingsläufe

```bash
# 1. Kontrolle: das Hauptmodell der Etappe 4 auf denselben 10 000 Samples, jetzt auf der GPU
cluster/maxwell/submit.sh train learned/spectrum-cnn-wide configs/learned/spectrum-cnn-wide.yaml \
    --train cpu-4x/train.h5 cpu-4x-extra/train.h5 --val cpu-4x/val.h5 --test cpu-4x/test.h5

# 2. Ensemble aus 5 Seeds auf 60 000 Samples (Laborbuch 04, 9.3): Task k trainiert mit --seed k
#    nach learned/spectrum-cnn-wide-60k-seedk
cluster/maxwell/submit.sh --array 0-4 train learned/spectrum-cnn-wide-60k \
    configs/learned/spectrum-cnn-wide.yaml \
    --train cpu-4x/train.h5 cpu-4x-extra/train.h5 cpu-4x-large/train.h5 \
    --val cpu-4x/val.h5 --test cpu-4x/test.h5

# 3. Bild-CNN auf 60 000 Samples (Laborbuch 04, 8.1 und 9.5)
cluster/maxwell/submit.sh train learned/image-cnn-60k configs/learned/image-cnn.yaml \
    --train cpu-4x/train.h5 cpu-4x-extra/train.h5 cpu-4x-large/train.h5 \
    --val cpu-4x/val.h5 --test cpu-4x/test.h5
```

Jeder Job trainiert, wählt die Epoche mit dem kleinsten Val-MAE, kalibriert auf Val und
schreibt die Vorhersagen für Val und Test nach `<lauf>/val/` und `<lauf>/test/`. Weitere
Optionen wie `--limit N` reicht er an `holofocus-learn train` weiter. Seed 0 ist der Seed des
Hauptmodells.

Das Training auf der GPU wird nicht bitgleich mit dem auf der CPU sein (andere Kernel und
Summationsreihenfolge). Lauf 1 sollte deshalb ähnliche, nicht dieselben Zahlen liefern wie
`results/learned/spectrum-cnn-wide/summary.json`: Val-MAE 190 µm, 42 % innerhalb 100 µm,
beste Epoche 67 von 100. Ähnlich heißt innerhalb der Streuung zwischen Seeds; bei 2000
Samples lagen drei Seeds zwischen 272 und 307 µm Val-MAE, also etwa ±6 % (Laborbuch 04, 5.2).
Vergleich:

```bash
cluster/maxwell/submit.sh --local run holofocus-learn summary "$R/learned/spectrum-cnn-wide"
```

Auf der VM brauchte Lauf 1 auf der CPU mit einem Thread 84 s zum Laden und 2258 s zum
Training; die Zeiten auf der A100 stehen im Log und in `run.yaml`. Die Eingaben liegen im
Arbeitsspeicher, die Hologramme selbst nicht: beim Spektrum 4 KB pro Sample (60 000 Samples
0,25 GB), beim Bild-CNN 128 KB (float16, 256²; 60 000 Samples 7,9 GB). Reicht der Speicher
eines Jobs nicht (Abschnitt 2), `--sbatch --mem=32G` vor `train` setzen. Ein Training, das am
Zeitlimit abbricht, beginnt mit demselben Befehl von vorn, weil das Modell erst am Ende
gespeichert wird.

### 7.1 Etappe 5: Modelle für die vier Messhologramme (Laborbuch 05)

Etappe 5 trainiert für jedes der vier Zenodo-Hologramme ein eigenes Spektrum-CNN, auf Daten
mit dessen Energie, Geometrie und Detektorpfad: `configs/dataset/matched-<probe>.yaml`, wobei
`<probe>` für `spider_hair`, `tooth`, `cactus_needle` oder `magnesium_wire` steht. Auf der VM
waren es 6000 Trainings-Samples pro Probe. `matched-<probe>-large.yaml` legt 50 000 weitere
mit eigenem Seed und derselben Physik dazu (das prüft ein Test in
`tests/test_dataset_config.py`); Val und Test bleiben die 250 Samples der kleinen Config. Die
realistischen Configs (`matched-cactus_needle-realistic.yaml`,
`matched-magnesium_wire-realistic.yaml`) gehören nicht dazu; sie warten auf ein besseres
Objektmodell (Laborbuch 05, 5.7).

| Probe | Simulationsgitter | VM, 3 Threads | 50 000 Samples |
|---|---|---|---|
| `spider_hair` | 4480² | 0,66 s/Sample | 34 GB |
| `tooth` | 3087² | 0,31 s/Sample | 29 GB |
| `cactus_needle` | 1620² | 0,14 s/Sample | 33 GB |
| `magnesium_wire` | 1568² | 0,12 s/Sample | 33 GB |

```bash
for p in spider_hair tooth cactus_needle magnesium_wire; do
    cluster/maxwell/submit.sh --cpu datagen configs/dataset/matched-$p.yaml --splits val test
    cluster/maxwell/submit.sh datagen configs/dataset/matched-$p.yaml --splits train
    cluster/maxwell/submit.sh --time 12:00:00 datagen configs/dataset/matched-$p-large.yaml
done
```

Es gelten dieselben Regeln wie in Abschnitt 5: Val und Test mit `--cpu`, damit der Job sie mit
den Fingerabdrücken `results/datasets/matched-<probe>-{val,test}.csv` vergleicht (erwartet
PASS oder WARN), und die kleine Config auf der GPU nur mit `--splits train`. Die Zeit pro
Sample wächst ungefähr mit der Pixelzahl des Simulationsgitters. Der gpu-check misst sie für
`cpu-4x` (3840²); das Spinnenhaar braucht etwa das 1,4-Fache davon, der Zahn das 0,65-Fache,
Kaktusnadel und Mg-Draht weniger als ein Fünftel. Mit diesen Faktoren prüfst du vor dem
Abschicken, ob 50 000 Samples ins Zeitlimit passen.

Platz: Mit den kleinen Configs (3,8–4,4 GB pro Probe) brauchen die vier Proben etwa 150 GB auf
Scratch, zusätzlich zu den rund 50 GB aus Abschnitt 5. Vorher mit `df -h` prüfen (Abschnitt 2).
Jeder Datensatz lässt sich aus seiner Config neu erzeugen; was nach dem Training nicht mehr
gebraucht wird, darf gelöscht werden.

Training, wenn alle Daten da sind (mit `--after` auch direkt hinter den Datenjobs):

```bash
for p in spider_hair tooth cactus_needle magnesium_wire; do
    cluster/maxwell/submit.sh train learned-large/matched-$p \
        configs/learned/spectrum-cnn-wide.yaml \
        --train matched-$p/train.h5 matched-$p-large/train.h5 \
        --val matched-$p/val.h5 --test matched-$p/test.h5
done
# Kaktusnadel und Mg-Draht zusätzlich mit dem Spektrum bis f^2 = 0,25 (Laborbuch 05, Abschnitt 3)
for p in cactus_needle magnesium_wire; do
    cluster/maxwell/submit.sh train learned-large/matched-$p-hf \
        configs/learned/spectrum-cnn-wide-hf.yaml \
        --train matched-$p/train.h5 matched-$p-large/train.h5 \
        --val matched-$p/val.h5 --test matched-$p/test.h5
done
```

Die Namen `learned-large/matched-<probe>` und `learned-large/matched-<probe>-hf` nicht ändern:
Daran erkennt `scripts/run_stage5_study.py` die Varianten einer Probe und nimmt die mit dem
kleineren Val-MAE. Abschnitt 9 nimmt `$R/learned-large` mit den Modellen mit zurück.

Die Schätzung auf den Messhologrammen dauert pro Modell Millisekunden und läuft auf jedem
Rechner mit dem Repo und den Hologrammen (`Examples/data/holograms` aus dem Zenodo-Archiv
10.5281/zenodo.8349365, Laborbuch 03), z. B. auf der VM nach dem Zurückgeben:

```bash
L=results/maxwell/learned-large
M=results/maxwell/measured-large
python scripts/run_stage5_study.py --learned $L measured --holograms DIR --out $M/cnn
python scripts/run_stage5_study.py --learned $L compare --cnn $M/cnn --out $M
python scripts/make_stage5_figures.py --learned $L --measured $M --sim2real $M --out $M/figures
```

`compare` nimmt die Baselines aus `results/measured` (Etappe 3 und 5), so dass
`$M/comparison.csv` dieselben Spalten hat wie `results/measured/comparison.csv` mit den Modellen
der VM. In `$M/figures` entstehen `stage5-measured.png`, `stage5-sim2real.png` und
`stage5-training.png` für die neuen Modelle; die übrigen Bilder brauchen Eingaben der VM und
fehlen.

## 8 NPE und weitere Läufe

### 8.1 Etappe 7: NPE (Laborbuch 07)

Etappe 7 schätzt die ganze Posterior von `dz01` mit neuronaler Posterior-Schätzung (NPE, das
Paket sbi) statt eines Mittelwerts mit σ. Die Umgebung braucht dafür sbi:
`cluster/maxwell/setup_env.sh --with-sbi` installiert das Extra `npe` des Pakets mit sbi 0.27.0
und dessen Abhängigkeiten in den Versionen der VM (`requirements-vm.txt`); torch und die
übrigen gepinnten Pakete bleiben unverändert. Bei `HOLOFOCUS_ENV_KIND=none` stattdessen
`pip install -c cluster/maxwell/requirements-vm.txt -e ".[dev,npe]"`. `holofocus.learned.npe`
nutzt interne Funktionen von sbi 0.27; mit einer anderen sbi-Version ist nichts geprüft.

Der Job `npe` trainiert mit `holofocus-learn npe-train`, behält die Epoche mit dem kleinsten
Val-Verlust (−log q des wahren `dz01`), sagt Val vorher, mit `--test` den Test-Split und mit
`--calibration` den Kalibrier-Split; auf diesem rechnet er auch die Prüfungen von sbi (SBC,
erwartete Abdeckung und TARP, `calibration/sbi-checks.json`). Als Array trainiert Task k mit
`--seed k` nach `<name>-seedk`.

Zuerst der Kalibrier-Split (2000 Ziehungen aus dem Prior mit der Physik von cpu-4x, eigener
Seed), auf der CPU, damit der Job ihn mit seinem Fingerabdruck
`results/datasets/cpu-4x-calibration-calibration.csv` vergleicht (PASS oder WARN wie in
Abschnitt 5). Auf der VM dauerte er mit 3 Threads 958 s (0,48 s pro Sample) und ist 1,4 GB
groß.

```bash
cluster/maxwell/submit.sh --cpu --time 08:00:00 datagen configs/dataset/cpu-4x-calibration.yaml
```

Dann die Läufe. Die Namen unter `npe/` und `npe-large/` folgen `results/npe/` der VM (Seed 0
ohne Endung, die Seeds 1–4 als Array), damit `scripts/run_stage7_study.py` sie unverändert
auswertet:

```bash
# 1. Kontrolle: der NPE der VM auf denselben 10 000 Samples, jetzt auf der GPU
cluster/maxwell/submit.sh npe npe/npe-nsf configs/learned/npe-nsf.yaml \
    --train cpu-4x/train.h5 cpu-4x-extra/train.h5 --val cpu-4x/val.h5 --test cpu-4x/test.h5 \
    --calibration cpu-4x-calibration/calibration.h5

# 2. 60 000 Samples (cpu-4x-large aus Abschnitt 5): Seed 0, die Seeds 1-4, die Mischdichte
large=(--train cpu-4x/train.h5 cpu-4x-extra/train.h5 cpu-4x-large/train.h5 --val cpu-4x/val.h5
    --test cpu-4x/test.h5 --calibration cpu-4x-calibration/calibration.h5)
cluster/maxwell/submit.sh npe npe-large/npe-nsf configs/learned/npe-nsf.yaml "${large[@]}"
cluster/maxwell/submit.sh --array 1-4 npe npe-large/npe-nsf configs/learned/npe-nsf.yaml \
    "${large[@]}"
cluster/maxwell/submit.sh npe npe-large/npe-mdn configs/learned/npe-mdn.yaml "${large[@]}"

# 3. die vier Messproben auf 56 000 Samples (Daten aus 7.1)
for p in spider_hair tooth cactus_needle magnesium_wire; do
    cluster/maxwell/submit.sh npe npe-large/matched-$p configs/learned/npe-nsf.yaml \
        --train matched-$p/train.h5 matched-$p-large/train.h5 \
        --val matched-$p/val.h5 --test matched-$p/test.h5
done
```

Lauf 1 sollte ähnliche, nicht dieselben Zahlen liefern wie `results/npe/npe-nsf/summary.json`
(auf der VM: beste Epoche 27, Val-MAE 196 µm, Abdeckung der 68-/95-%-Intervalle 0,68/0,92,
48 min Training mit 1 Thread); der Grund steht in Abschnitt 7. Auf der VM passten sich alle
NPE-Läufe nach 21 bis 38 von 100 Epochen den Trainingsdaten an: Der Val-Verlust stieg danach
wieder, während der Trainingsverlust weiter fiel (Laborbuch 07). Die Auswahl nach dem
Val-Verlust fängt das ab, die übrigen Epochen kosten nur Zeit. Ob 60 000 Samples die beste
Epoche nach hinten schieben, zeigt `history.csv` von Lauf 2.

Ausgewertet wird nach dem Zurückgeben (Abschnitt 9) auf der VM, mit den CNNs auf denselben
Daten als Vergleich (Abschnitt 7, Lauf 2, und Abschnitt 7.1):

```bash
N=results/maxwell/npe-large
C=results/maxwell/learned/spectrum-cnn-wide-60k-seed0
S="python scripts/run_stage7_study.py --npe $N --cnn $C"
$S --threads 3 predict
$S compare
$S --threads 1 timing
$S examples
$S --threads 3 perturb --cnn-learned results/maxwell/learned-large --cnn-sim2real $N/cnn-sim2real
$S --threads 1 measured --holograms DIR
python scripts/make_stage7_figures.py --npe $N --cnn $C --learned results/maxwell/learned-large \
    --out $N/figures
```

`perturb` vergleicht mit den Vorhersagen des CNN derselben Probe auf denselben gestörten Sets
und nimmt sie nur, wenn Datensatz und Modell (SHA-256) mit der Referenz übereinstimmen. Für die
großen CNNs erzeugt sie
`python scripts/run_stage5_study.py --learned results/maxwell/learned-large --threads 3 perturb --out $N/cnn-sim2real`
(Laborbuch 05); fehlen sie, bleibt die Spalte des CNN in `$N/sim2real/<probe>.json` leer.
Die Zeit pro Schätzung (`timing`) gehört auf die CPU der VM wie in Etappe 4, nicht auf die GPU.

Nur auf Maxwell zu prüfen (**PRÜFEN**): ob der NPE auf der A100 trainiert und vorhersagt (der
CUDA-Pfad von sbi und des Gitters ist auf der VM nicht gelaufen), die Zeit pro Epoche, der
Speicher (die Eingaben brauchen 4 KB pro Sample wie beim Spektrum-CNN) und ob die Zeitlimits
reichen. Die Batchgröße 32 und die übrigen Einstellungen der Config bleiben wie auf der VM,
damit die Läufe vergleichbar sind.

### 8.2 Dichter MFE-Scan (Laborbuch 04, 9.1)

Woher die schmalen Senken des MFE kommen, soll ein dichter Scan klären: der MFE eines Samples
in Schritten von 10 µm, einmal mit der Iterationszahl von [2] und einmal mit der doppelten. Das
Skript rechnet nur auf der CPU, deshalb `--cpu`:

```bash
cluster/maxwell/submit.sh --cpu run bash -c 'python scripts/run_stage3_experiments.py \
    bias --data "$HOLOFOCUS_DATA/cpu-4x/test.h5" --indices 35 --variants noisy \
    --points 161 --half-range 8e-4 --out "$HOLOFOCUS_RUNS/mfe-scan/sample35"'
cluster/maxwell/submit.sh --cpu run bash -c 'python scripts/run_stage3_experiments.py \
    --iteration-scale 2 bias --data "$HOLOFOCUS_DATA/cpu-4x/test.h5" --indices 35 \
    --variants noisy --points 161 --half-range 8e-4 --out "$HOLOFOCUS_RUNS/mfe-scan/sample35-it2"'
```

Das sind 161 Punkte über ±0,8 mm um das wahre z01; statt Sample 35 gehen auch 27 oder 41. Auf
der VM wären es 161 × 29 s ≈ 80 min mit 3 Threads, mit doppelter Iterationszahl etwa das
Doppelte. Mit nur einer CPU (Zeile `host` im Log) dauert es etwa dreimal so lange; dann
`--sbatch --cpus-per-task=8` angeben. Das Ergebnis steht in `bias-35.json`; `run` kopiert
nichts nach DUST.

## 9 Ergebnisse sichern und zurückgeben

Jeder Job außer `run` kopiert am Ende sein Laufverzeichnis ohne HDF5-Dateien nach DUST, an
dieselbe Stelle relativ zu `$HOLOFOCUS_DUST/holofocus` wie unter `$HOLOFOCUS_SCRATCH`
(z. B. `$HOLOFOCUS_DUST/holofocus/runs/baseline/cpu-4x-test-p2/chunk-3-of-10`). Ist
`HOLOFOCUS_DUST=""`, bleibt alles nur auf Scratch.

Ins Repo gehören die kleinen Dateien, auf die sich das Laborbuch stützt: die zusammengeführten
Baseline-Läufe ohne ihre Stücke, die Vergleiche, die Trainingsläufe mit ihren Modellen
(`model.pt`, 1–2 MB), die Vorhersagen, die gpu-checks, die MFE-Scans, die Logs und die
Versionsliste der Umgebung. So geht es zurück:

```bash
source cluster/maxwell/config.sh
R=$HOLOFOCUS_SCRATCH/runs
M=results/maxwell
git checkout -b maxwell-ergebnisse-$(date +%Y%m%d)
mkdir -p $M/logs
cp -r $R/baseline $R/comparison $R/learned $R/learned-large $R/gpu-check $R/mfe-scan $M/  # was es davon gibt
cp -r $R/npe $R/npe-large $M/       # die NPE-Läufe aus 8.1, falls trainiert
rm -rf $M/baseline/*/chunk-*
cp $HOLOFOCUS_SCRATCH/logs/holofocus-*.out $M/logs/
cp $HOLOFOCUS_ENV/holofocus-env.txt $M/
git add $M                          # HDF5-Dateien und *.pt schließt .gitignore aus
git add -f $M/learned/*/model.pt    # die trainierten Modelle gehören dazu
git add -f $M/learned-large/*/model.pt   # die Modelle aus 7.1, falls trainiert
git add -f $M/npe*/*/model.pt       # die NPEs aus 8.1, falls trainiert
git commit -m "Maxwell: Baseline auf dem ganzen Test-Split, Training auf 60 000 Samples"
git push -u origin maxwell-ergebnisse-$(date +%Y%m%d)
```

Den Branch-Namen gibst du an den Koordinator weiter; die nächste Etappe übernimmt die
Ergebnisse von dort ins Laborbuch. Fragt Git nach Name und E-Mail, einmal
`git config --global user.name "..."` und `git config --global user.email "..."` setzen. Geht
von Maxwell aus kein Push zu GitHub (kein SSH-Schlüssel oder Token), stattdessen
`tar czf maxwell-ergebnisse.tgz results/maxwell` und die Datei auf einen Rechner mit Zugang
kopieren.

## 10 Fehlersuche

| Symptom | Ursache und Abhilfe |
|---|---|
| `replace CHECK_ME in ... first` | in `config.sh` steht noch ein `CHECK_ME`; die Meldung nennt die Einstellungen |
| `no settings file ...` | `cluster/maxwell/config.sh` fehlt (Abschnitt 3) |
| Job bleibt in `PD` | `squeue -u $USER -o "%.10i %.25j %.8T %.25R"` zeigt den Grund: `Resources` oder `Priority` heißt warten; `PartitionTimeLimit` heißt `--time` kleiner wählen; `DependencyNeverSatisfied` heißt, ein Job, auf den er wartet, ist gescheitert (6.4). Genaueres mit `scontrol show job <id>` |
| `CUDA is not available on <knoten>` | der Knoten hat keine GPU für den Job (Partition, Constraint, `HOLOFOCUS_GPU_REQUEST` prüfen) oder der Treiber ist älter als die CUDA-Version der Wheels (`nvidia-smi` gegen `HOLOFOCUS_TORCH_INDEX`, dann `setup_env.sh --recreate`) |
| Treiber zu alt für `cu126` | ältere PyTorch-Version: `HOLOFOCUS_TORCH_SPEC="torch==2.5.1"` mit dem Index `.../whl/cu121` (Treiber mit CUDA 12.x) oder `.../whl/cu118` (CUDA 11.x), dann `setup_env.sh --recreate` und den gpu-check wiederholen. Mit torch 2.5.1 (CPU) bestehen auf der VM alle Tests; die Ergebnisse können in den letzten Stellen abweichen (WARN) |
| `is older than Python 3.12` | numpy 2.5.4 und scipy 1.18.1 der VM brauchen Python 3.12: ein neueres Python-Modul laden oder `HOLOFOCUS_ENV_KIND="conda"` (Abschnitt 3) |
| `No matching distribution found for torch==...` | glibc unter 2.28 oder eine Python-Version ohne Wheel; conda mit Python 3.12 nehmen |
| `Disk quota exceeded` oder `No space left on device` beim Einrichten | `df -h ~ /tmp $HOLOFOCUS_SCRATCH` zeigt, wo es eng ist. pip entpackt in `$TMPDIR` (meist `/tmp`): `mkdir -p $HOLOFOCUS_SCRATCH/tmp`, dann `TMPDIR=$HOLOFOCUS_SCRATCH/tmp cluster/maxwell/setup_env.sh` |
| `module: command not found` | `HOLOFOCUS_MODULES=""` setzen oder klären, wie Module in Batch-Jobs geladen werden |
| `DUE TO TIME LIMIT` im Log | denselben Befehl noch einmal, bei einem Array mit `--array k` nur für das Stück: die Baseline setzt fort, ein Training beginnt neu. Sonst `--time` größer oder mehr Stücke |
| `already holds a trained model` | der Lauf ist schon trainiert, nur Vorhersagen fehlen: `cluster/maxwell/submit.sh predict "$R/learned/<lauf>" cpu-4x/test.h5 learned/<lauf>/test` (für Val entsprechend); ein neues Training in ein anderes Verzeichnis |
| Speicherfehler (`oom-kill`, `Out Of Memory`) | mehr Arbeitsspeicher mit `--sbatch --mem=...` (Abschnitte 2 und 7) |
| `a job array needs --chunks N` | ein Baseline-Array ohne `--chunks`: `--chunks N` passend zu `--array 0-(N-1)` angeben |
| `--chunks needs a job array` | `--chunks` ohne `--array`: `--array 0-(N-1)` ergänzen oder `--chunks` weglassen (dann ein Job für alle Samples) |
| `holds a run with another config or dataset` | im Laufverzeichnis liegt ein Lauf mit anderer Config, anderen Daten oder anderem Warmstart: anderen Namen wählen oder das alte Verzeichnis löschen |
| `hold runs with another config, dataset or warm start` | `merge` hat in `<name>/chunk-*` Stücke verschiedener Läufe gefunden; die falschen löschen |
| `sample i is in ... and ...` | `merge` hat dasselbe Sample in zwei Stücken gefunden, z. B. nach Läufen mit verschiedenem `--chunks`: die Stücke mit dem falschen `-of-N` löschen und `merge` wiederholen |
| `keep: copying ... failed (DUST quota? ...)` | die Kopie nach DUST ist gescheitert, meist ist das Kontingent voll; die Ergebnisse liegen weiter auf Scratch. Platz schaffen und den Job wiederholen (eine fertige Baseline rechnet dabei nichts neu) oder das Verzeichnis selbst kopieren; `HOLOFOCUS_DUST=""` schaltet die Kopie ab |
| `data labels` FAIL | andere numpy-Version oder geänderte Config: Split nicht mit den committeten Ergebnissen vergleichen, `holofocus-env.txt` und `git log` prüfen |
| `baseline` oder `predictions` FAIL | es wird anders gerechnet (andere Config, TF32, andere Daten); Log an den Koordinator |

## 11 Was auf der VM geprüft ist und was nicht

Auf der VM geprüft (4 CPU-Kerne, ohne SLURM und ohne GPU, mit `--device cpu`):

- Alle Skripte mit `bash -n` und shellcheck, ohne Befund; beides ist Teil der Testsuite.
- `submit.sh`: der sbatch-Befehl im `--dry-run` mit Partition, Constraint, GPU-Anforderung,
  Konto, Array, Zeitlimit, Abhängigkeit und `--export`; die Ausgabe der Job-ID mit einem
  nachgebauten `sbatch --parsable`; die Ablehnung von `CHECK_ME`, unbekannten Jobs, einem
  Baseline-Array ohne `--chunks` und `--chunks` ohne Array (alles Tests).
- `setup_env.sh` als venv mit den CPU-Wheels von torch 2.14.1 (34 s) und mit conda aus einem
  frischen Miniforge nach Abschnitt 3 (Python 3.12.15, 34 s): Die Downloads landen in
  `$HOLOFOCUS_SCRATCH/cache`, `~/.cache/pip` bleibt unverändert. `--with-sbi` installiert
  das Extra `npe` mit sbi 0.27.0 und lässt torch 2.14.1 und numpy 2.5.4 unverändert
  (venv mit den CPU-Wheels, 40 s; zuko, nflows, scikit-learn und die übrigen Abhängigkeiten in
  den Versionen der VM). Ein Python älter als 3.12 wird abgelehnt (Test).
- Die Aktivierung der Umgebung im Job für venv und conda, mit einem minimalen `PATH`.
- Alle Jobs mit `--local` auf dem CI-Datensatz: `datagen`, `train`, `baseline` als Array mit
  zwei Stücken in umgekehrter Reihenfolge, `merge` und die Kopie nach DUST ohne HDF5-Dateien
  (Test, 24 s); `npe` als Array-Task 1 mit Training nach `<name>-seed1` und Seed 1 in der
  Config, Vorhersagen für Val und den Kalibrier-Split, den Prüfungen von sbi und der Kopie nach
  DUST (Test, 13 s).
- `gpu-check --local` auf der CPU der VM: alle Prüfungen PASS einschließlich `training`, „no
  check failed“, 5,5 min mit 4 Threads; die Baseline von Sample 0 trifft die Schätzung aus
  Etappe 3 auf 0 µm.
- Die Befehle aus 6.4, 6.5 und 8.2 mit `--local run`: `check` gibt für eine Kopie des
  Warmstart-Laufs der VM PASS mit 0 µm Abweichung auf 48 Samples; `compare` erzeugt aus den
  committeten Läufen `results/comparison/cpu-4x-test/` bytegleich; der MFE-Scan läuft als
  `run bash -c '...'` (mit 3 Punkten und 2 % der Iterationen), und `$HOLOFOCUS_DATA` und
  `$HOLOFOCUS_RUNS` werden im Job aufgelöst.
- torch 2.5.1 (CPU) mit den übrigen Versionen der VM: 216 Tests bestehen, auch die der Jobs
  mit `--local`; nur der CUDA-Test wird ohne GPU übersprungen, wie mit torch 2.14.1
  (Ausweichlösung bei einem alten Treiber, Abschnitt 10).
- Die Fingerabdrücke `results/datasets/cpu-4x-val.csv` und `cpu-4x-test.csv` gegen die Splits
  der VM: PASS.
- Drei gpu-checks auf derselben CPU ergaben für dieselbe Rekonstruktion drei MFE-Werte, die
  sich um bis zu 1,1·10⁻⁴ relativ unterschieden: Die ASRM auf der CPU ist zwischen zwei Läufen
  nicht bitgleich. Die Schätzung von Sample 0 war trotzdem jedes Mal dieselbe. Abweichungen
  dieser Größe sind also keine Besonderheit der GPU; die Schwelle des Vergleichs mit der CPU
  im gpu-check liegt bei 10⁻³.

Nur auf Maxwell zu prüfen (**PRÜFEN**, Abschnitt 2 und gpu-check):

- SLURM selbst: Namen von Partitionen und Constraint, die Form der GPU-Anforderung, Konto,
  Zeitlimits, CPUs und Speicher ohne Angabe, ganze oder geteilte Knoten.
- Module und Python-Version, ob `module load` in Batch-Jobs funktioniert (der Kit sucht die
  üblichen Init-Skripte), Internetzugang und glibc des Login-Knotens.
- Treiber und CUDA-Version der A100-Knoten und damit der PyTorch-Index.
- Alles, was auf der GPU rechnet: die Ergebnisse des gpu-check auf der A100 (Abschnitt 4), die
  Laufzeiten und damit die Zahl der Stücke und die Zeitlimits, der NPE auf der GPU
  (Abschnitt 8.1).
- Pfade und Kontingente von Scratch und DUST und ob alte Dateien auf Scratch gelöscht werden.
- Ob von Maxwell aus ein Push zu GitHub geht (Abschnitt 9).
