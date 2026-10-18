# Kaggle runs

Procedure to run the missing arms of this project under your own Kaggle account with the `kaggle` CLI
(tested with Kaggle CLI 2.2.4).

## Get the code

    git clone -c core.autocrlf=false -b recursion_comparison --single-branch https://github.com/WinterBlossom0/CinnamonLM.git recursion_comparison
    cd recursion_comparison

Run every command below in this folder.

## What to run

Run exactly these 4 arms, one GPU kernel each: `grt_x8`, `parcae_x8`, `reinject_x8`, `mesh_x8`.

The other 7 arms (`recursive_x2`, `real_6`, `recursive_x4`, `real_10`, `recursive_x8`, `real_18`, `hyperloop_x8`)
are kept: their code path was checked identical to this code (same initial weights and bit-identical gradients
against the code saved by their Kaggle runs). Do not run them. Their metrics.json files come from runs made before
the code recorded `code_sha256`, so they have no such field.

Reruns are not bit-reproducible per seed: the compiled backward adds with atomics in a random order (about 1e-8
noise per step), and some arms are bimodal, so one seed can land in either of two loss clusters. Compare the mean
of the 3 seeds, not single seeds.

## Cost

- 1 CPU session for the data kernel (about 15 minutes, no GPU quota).
- 4 GPU sessions, one per arm, each on 2x T4. Each runs 3 seeds of 3 epochs, 30,114 steps per seed: seeds 0 and 1
  in parallel (one GPU each, batch 640), then seed 2 split over both GPUs (batch 320 each).
- Time per GPU session, estimate from measured step times. Measured on Kaggle 2x T4: a training step takes 115-131 ms
  at batch 640 and 59-67 ms at batch 320; the checks took about 13 min for 4 arms (about 3 min per arm); setup
  (pip) took 4.4 min; loading the tokens and compiling took about 4 min per session.

  | Kernel | Step ms (640 / 320) | Setup, check, loading | Seeds 0 and 1 | Seed 2 split | Total |
  |---|---|---|---|---|---|
  | `grt_x8` | 129.1 / 65.6 | 12 min | 65 min | 33 min | 1.8 h |
  | `parcae_x8` | 118.2 / 60.0 | 12 min | 59 min | 30 min | 1.7 h |
  | `reinject_x8` | 115.5 / 58.8 | 12 min | 58 min | 30 min | 1.7 h |
  | `mesh_x8` | 131.2 / 66.7 | 12 min | 66 min | 33 min | 1.9 h |

  Total for the 4 kernels: about 7 GPU hours. In the older Kaggle runs of `grt_x8`, `mesh_x8` and `hyperloop_x8`,
  whole training runs took 132-155 ms per step at batch 640; at that speed a kernel takes up to about 2.2 h, and
  the 4 kernels about 8.6 h.
- A session uses GPU quota for its full wall time. The weekly GPU quota is 30 h. `kaggle quota` shows what is left.
- Kaggle runs at most 2 GPU sessions at the same time.

## Requirements

- `pip install kaggle`, and Python 3 for `push.py build` (standard library only).
- A phone-verified Kaggle account: the kernels need internet (pip and the TinyStories download).
- Kaggle credentials, in one of these forms. The CLI tries them in this order:
  1. An API token from https://www.kaggle.com/settings/api ("Generate New Token"), in the environment variable
     `KAGGLE_API_TOKEN` or in the file `~/.kaggle/access_token`.
  2. A legacy key: the file `~/.kaggle/kaggle.json` (`{"username": "...", "key": "..."}`), or the environment
     variables `KAGGLE_USERNAME` and `KAGGLE_KEY`. `KAGGLE_CONFIG_DIR` moves the folder of `kaggle.json`.
  3. OAuth: `kaggle auth login` (add `--no-launch-browser` on a machine without a browser).

  `kaggle config view` shows the username in use.

## Names

`OWNER` is your Kaggle username. All names are fixed; do not rename them.

| Kind | Name | Purpose |
|---|---|---|
| dataset | `OWNER/recursion-comparison-src` | the source files |
| CPU kernel | `OWNER/recursion-comparison-data` | TinyStories packed into tokens; run once, before any GPU kernel |
| GPU kernel | `OWNER/recursion-comparison-<arm>` | one arm, 3 seeds; `<arm>` with `-` for `_`, e.g. `grt-x8` for `grt_x8` |
| GPU kernel | `OWNER/recursion-comparison-smoke` | the checks and a shortened run of the 4 arms; optional, 1 more GPU session (51 min with earlier code) |

## Steps

1. Build the upload folders for your account:

       python push.py build --owner OWNER

2. Upload the source dataset. Wait until the status says `ready`:

       kaggle datasets create -p kaggle_build/src -r zip
       kaggle datasets status OWNER/recursion-comparison-src

   If the dataset exists already, use `kaggle datasets version -p kaggle_build/src -m "update" -r zip`.

3. Run the data kernel. Wait until the status says `COMPLETE`:

       kaggle kernels push -p kaggle_build/kernels/recursion-comparison-data
       kaggle kernels status OWNER/recursion-comparison-data

4. Run the 4 arm kernels, at most 2 at a time. Push the next one when one ends:

       kaggle kernels push -p kaggle_build/kernels/recursion-comparison-grt-x8
       kaggle kernels push -p kaggle_build/kernels/recursion-comparison-parcae-x8
       kaggle kernels push -p kaggle_build/kernels/recursion-comparison-reinject-x8
       kaggle kernels push -p kaggle_build/kernels/recursion-comparison-mesh-x8
       kaggle kernels status OWNER/recursion-comparison-grt-x8

   The final states are `COMPLETE`, `ERROR` and `CANCEL_ACKNOWLEDGED`. Check the status every minute or less often.
   Do not push a kernel again while it runs: the new version runs beside the old one and uses quota too.
   A kernel first runs `check.py` for its arm; a failed check stops it before training.

5. Download the results when the status says `COMPLETE`, into the git-ignored folder `kaggle_output/<arm>`:

       kaggle kernels output OWNER/recursion-comparison-grt-x8 -p kaggle_output/grt_x8 --file-pattern ".*(metrics\.json|model\.pt)$"

   `python push.py pull grt_x8 --owner OWNER` does the same and warns for each metrics.json whose `code_sha256`
   differs from the hash of the local code.

   The files are `src/results/<arm>/seed_<n>/metrics.json` and `model.pt` (the final state_dict) next to it (the
   smoke kernel writes `src/results_smoke/...`). Each metrics.json records the final eval loss, the history, the
   SHA-256 of the code and of the data, the package versions and the GPUs. The same command also writes the kernel
   log to `kaggle_output/grt_x8/recursion-comparison-grt-x8.log`.

6. Check that each run used this code. The expected `code_sha256` is

       da7e6361bd54cc8d589f614ea334de8da81f66bb9e222ddb7e478e5b60a7fe66

   This command prints each downloaded metrics.json with another hash; no output means that all match:

       grep -L '"code_sha256": "da7e6361bd54cc8d589f614ea334de8da81f66bb9e222ddb7e478e5b60a7fe66"' kaggle_output/*/src/results/*/seed_*/metrics.json

   Do not use a run that does not match: its code is not this code. The hash is over the exact bytes of the files,
   so a checkout with CRLF line endings gives another hash; clone with `git config core.autocrlf false`.

## Share the runs

The kernels are private. Share each of the 4 arm kernels with the Kaggle user `electroknight`:
open `https://www.kaggle.com/code/OWNER/recursion-comparison-<arm>`, click Share, add `electroknight` as a
collaborator, and save. The CLI has no command for this; a person must do it in the browser.

## Failures

If the status is `ERROR`, print the log of the latest run:

    kaggle kernels logs OWNER/recursion-comparison-<arm>

The log is empty while the kernel runs; `kaggle kernels logs -f OWNER/recursion-comparison-<arm>` streams it live.
Find these messages in the log:

- `packed tokens missing`: the data kernel did not finish, or the GPU kernel does not list it as a source; run step 3.
- `token SHA-256 ... expected ...`: the data differs from the data of the reported runs; do not use the results.
- `source dataset not attached`: the dataset version was not ready when the kernel started; push the kernel again.
- `Maximum batch GPU session count of 2 reached` (from `kaggle kernels push`): wait until a GPU kernel ends.
- `FAIL: <message>` from `check.py`, followed by `CalledProcessError`: the arm failed its checks on this GPU, so the
  kernel stopped before training. Do not push the kernel again; send the log to the project owner.
