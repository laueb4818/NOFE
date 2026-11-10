#!/usr/bin/env bash
# Install the R dependencies required by ERA5/notebooks/execute_spatialpca.R, which
# ERA5/notebooks/2d_scatter.ipynb calls via `conda run -n spatialpca Rscript ...`.
#
# Why this is needed at all: SpatialPCA is NOT on CRAN (it has been archived, so
# `install.packages("SpatialPCA")` fails with "not available for this version of R") and
# it has no conda-forge/bioconda package. It is only installable from GitHub. Without it
# the notebook dies at:
#     Error in library(SpatialPCA) : there is no package called 'SpatialPCA'
#
# Install method note: upstream's README says `remotes::install_github(...)`, but that
# goes through the GitHub *API* and fails on machines where a `gh` credential helper is
# configured (the helper returns a token the API rejects). This script instead downloads
# the source tarballs from codeload.github.com with curl and installs them locally, which
# needs no git credentials and no GitHub API.
#
# Usage:
#     bash ERA5/spatialpca/install_spatialpca_R_deps.sh
#
# Run it from anywhere; it locates the `spatialpca` conda env. Idempotent: re-running
# skips anything already installed.

set -euo pipefail

ENV_NAME="${SPATIALPCA_ENV:-spatialpca}"

# --- locate the env's Rscript -------------------------------------------------
if [[ -n "${CONDA_PREFIX:-}" && "$(basename "$CONDA_PREFIX")" == "$ENV_NAME" ]]; then
    RSCRIPT="$CONDA_PREFIX/bin/Rscript"
else
    CONDA_BIN="$(command -v conda || true)"
    # Fall back to the usual install roots if conda is not on PATH.
    if [[ -z "$CONDA_BIN" ]]; then
        for c in "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" \
                 "$HOME/anaconda3/bin/conda" /opt/miniconda3/bin/conda /opt/anaconda3/bin/conda; do
            [[ -x "$c" ]] && { CONDA_BIN="$c"; break; }
        done
    fi
    if [[ ! -x "$CONDA_BIN" ]]; then
        echo "error: conda not found on PATH." >&2
        echo "       Either 'conda activate $ENV_NAME' first, or add conda to PATH." >&2
        exit 1
    fi
    # Ask conda where the env lives. `conda run` inherits CONDA_PREFIX from the parent
    # shell, so it can report the *base* prefix instead of the target env; query the env
    # list instead and match the name exactly.
    ENV_PREFIX="$("$CONDA_BIN" env list | awk -v n="$ENV_NAME" '$1==n {print $NF}' | tail -1)"
    if [[ -z "$ENV_PREFIX" ]]; then
        # fall back: assume the conventional location next to the conda binary
        ENV_PREFIX="$(dirname "$(dirname "$CONDA_BIN")")/envs/$ENV_NAME"
    fi
    RSCRIPT="$ENV_PREFIX/bin/Rscript"
fi

if [[ ! -x "$RSCRIPT" ]]; then
    echo "error: no Rscript at $RSCRIPT" >&2
    echo "       Create the env first: conda env create -f spatialpca_environment.yml" >&2
    exit 1
fi

# Neutralise R_LIBS_USER for everything below. R puts the *user* library
# (~/Library/R/arm64/<ver>/library on macOS) ahead of the env's own
# $CONDA_PREFIX/lib/R/library, and that user library typically holds CRAN macOS builds
# linked against the CRAN R framework rather than conda's libR. Letting those shadow the
# conda copies is what makes `library(SpatialPCA)` die in dyn.load with an unrecoverable
# segfault (SpatialPCA -> RSpectra -> Rcpp, and it is Rcpp's foreign .so that fails):
#     *** caught segfault ***   address 0x0, cause 'invalid permissions'
#     dyn.load(...) -> loadNamespace(...) -> namespaceImport(...) -> loadNamespace(...)
export R_LIBS_USER='/nonexistent-r-libs-user'
echo "R_LIBS_USER pinned to $R_LIBS_USER (prevents CRAN-built packages shadowing conda's)"

echo "Using: $RSCRIPT"
"$RSCRIPT" --version | head -1

# --- fetch sources ------------------------------------------------------------
CACHE="${TMPDIR:-/tmp}/spatialpca_src"
mkdir -p "$CACHE"
SPARK_TGZ="$CACHE/SPARK.tar.gz"
SPCA_TGZ="$CACHE/SpatialPCA.tar.gz"

fetch() {  # url dest
    if [[ -s "$2" ]]; then
        echo "  cached: $2"
    else
        echo "  downloading: $1"
        curl -fsSL --retry 3 -o "$2" "$1"
    fi
}
echo "== fetching sources"
fetch "https://codeload.github.com/xzhoulab/SPARK/tar.gz/refs/heads/master"      "$SPARK_TGZ"
fetch "https://codeload.github.com/shangll123/SpatialPCA/tar.gz/refs/heads/main" "$SPCA_TGZ"

# SPARK's repo ships stale x86-64 Linux object files (src/*.o, src/SPARK.so). R's build
# sees them and skips recompiling, then the link step produces an x86-64 ELF .so that
# fails to dlopen on arm64 macOS ("slice is not valid mach-o file"). Delete them from the
# extracted tree so SPARK is actually compiled for this machine. Only needed when SPARK
# is requested.
SPARK_DIR="$CACHE/SPARK-master"
if [[ ! -d "$SPARK_DIR" ]]; then
    echo "  extracting SPARK"
    tar xzf "$SPARK_TGZ" -C "$CACHE"
fi
if compgen -G "$SPARK_DIR/src/*.o" >/dev/null || [[ -f "$SPARK_DIR/src/SPARK.so" ]]; then
    echo "  removing stale prebuilt objects from SPARK/src"
    rm -f "$SPARK_DIR/src/"*.o "$SPARK_DIR/src/"*.so
fi

# SpatialPCA declares Seurat + SPARK as Imports purely for CreateSpatialPCAObject(), which
# execute_spatialpca.R never calls. R's install.packages() refuses to install a package
# whose Imports are unavailable, so both the Imports field and the matching import()
# directives are stripped from the extracted source. Without this the step fails with
#   ERROR: dependencies 'SPARK', 'Seurat', 'splatter' are not available for package 'SpatialPCA'
# The functions actually used (buildKernel / EstimateLoading / SpatialPCs) need only
# RSpectra + KernSmooth, so the resulting package loads and runs without Seurat/SPARK.
SPCA_DIR="$CACHE/SpatialPCA-main"
if [[ ! -d "$SPCA_DIR" ]]; then
    echo "  extracting SpatialPCA"
    tar xzf "$SPCA_TGZ" -C "$CACHE"
fi
if grep -q 'SPARK' "$SPCA_DIR/DESCRIPTION"; then
    echo "  dropping non-essential Imports/Remotes from SpatialPCA DESCRIPTION"
    # Imports is a continuation block ending at the next 'Field:' line, so a line-based
    # edit does not work here.
    python3 - "$SPCA_DIR/DESCRIPTION" <<'PYDESC'
import re, sys
p = sys.argv[1]
t = open(p).read()
t = re.sub(r'(?ms)^Imports:.*?(?=^[A-Za-z][A-Za-z0-9.]*:)',
           'Imports: methods, MASS, Matrix, RSpectra, parallel\n', t)
t = re.sub(r'(?m)^Remotes:.*$\n?', '', t)
open(p, 'w').write(t)
PYDESC
fi
if grep -qE '^import\((SPARK|Seurat|splatter|Rtsne|ggplot2|pdist|tidyr|umap)\)' "$SPCA_DIR/NAMESPACE"; then
    echo "  trimming SpatialPCA/NAMESPACE to what execute_spatialpca.R needs"
    python3 - "$SPCA_DIR/NAMESPACE" <<'PYNS'
import re, sys
p = sys.argv[1]
drop = re.compile(r'^import\((SPARK|Seurat|splatter|Rtsne|ggplot2|pdist|tidyr|umap)\)$')
lines = [l for l in open(p) if not drop.match(l.rstrip('\n'))]
open(p, 'w').writelines(lines)
PYNS
fi

# --- R packages ---------------------------------------------------------------
# The three functions the notebook actually calls
# (SpatialPCA_buildKernel / SpatialPCA_EstimateLoading / SpatialPCA_SpatialPCs) touch
# only RSpectra + KernSmooth, because execute_spatialpca.R deliberately bypasses
# CreateSpatialPCAObject() -- the only entry point that needs Seurat/SPARK. The default
# path here therefore installs just what the notebook needs; it is verified end-to-end
# (reproduces Siyan's committed spatialPCA numbers exactly) and needs no compilation.
#
# Set SPATIALPCA_WITH_SPARK=1 to additionally build SPARK + Seurat. That keeps the env
# faithful to upstream so the other SpatialPCA entry points work, but it is not required
# by this notebook, and SPARK's compiled .so was observed to fail dlopen on this machine
# ("slice is not valid mach-o file"). SpatialPCA's NAMESPACE imports Seurat, so SpatialPCA
# will not *load* without Seurat even though the notebook's code path never calls it --
# r-seurat from spatialpca_environment.yml provides it.
MINIMAL=1
[[ "${SPATIALPCA_WITH_SPARK:-0}" == "1" ]] && MINIMAL=0

# Quoted heredoc: bash performs NO expansion inside, so the R code is passed through
# byte-for-byte. (Embedding it in a double-quoted string made every `\.` and `"` a
# quoting hazard.) Config reaches R through the environment instead.
export SP_MINIMAL="$MINIMAL" SP_SPARK_DIR="$SPARK_DIR" SP_SPCA_DIR="$SPCA_DIR"
"$RSCRIPT" - <<'RSCRIPT'
options(repos = c(CRAN = 'https://cloud.r-project.org'),
        Ncpus = max(1L, parallel::detectCores() - 1L))
need <- function(p) requireNamespace(p, quietly = TRUE)
MINIMAL <- Sys.getenv('SP_MINIMAL')

# ---------------------------------------------------------------------------
# Do NOT install the compiled dependencies from CRAN -- they must come from conda.
# Two independent reasons:
#
#  1. CRAN macOS *binaries* link /Library/Frameworks/R.framework/.../libR.dylib (the CRAN
#     R runtime). This env is a conda r-base whose libR lives elsewhere. Loading such a
#     .so puts two libR copies in one process and dies at dyn.load:
#         *** caught segfault ***   address 0x0, cause 'invalid permissions'
#         ... dyn.load(file, ...) -> loadNamespace(...) -> namespaceImport(...)
#     That is exactly the failure a previous version of this script caused.
#
#  2. CRAN *source* builds are not possible either for the compiled ones: RSpectra links
#     -lgfortran and the R gfortran runtime (/opt/gfortran) is absent here, so it dies at
#         ld: library 'gfortran' not found
#
# So the compiled deps are conda's job -- spatialpca_environment.yml already lists
# r-matrix, r-rspectra, r-rtsne, r-pdist, r-umap, r-ggplot2, r-tidyr, r-dplyr,
# r-assertthat and r-seurat. Here we only CHECK those, and fetch the pure-R leftovers.
# ---------------------------------------------------------------------------
conda_provided <- c('MASS', 'Matrix', 'RSpectra', 'Rtsne', 'pdist', 'umap', 'tidyr',
                    'ggplot2', 'dplyr', 'assertthat')
pure_r <- c('KernSmooth')
if (MINIMAL != '1') {
    pure_r <- c(pure_r, 'Rcpp', 'foreach', 'doParallel', 'CompQuadForm', 'pracma', 'matlab')
}

# A package present but linked against a foreign libR segfaults on load, and
# requireNamespace() can still succeed for it -- so verify linkage and report loudly.
foreign <- character(0)
r_home <- normalizePath(R.home(), mustWork = FALSE)
for (p in c(conda_provided, pure_r)) {
    # find.package(quiet=TRUE) yields character(0) when absent, so index to one value
    # before testing (nzchar(character(0)) is logical(0), which breaks `if`).
    lib <- tryCatch(find.package(p, quiet = TRUE),
                    error = function(e) character(0))[1]
    if (is.na(lib) || !nzchar(lib)) next
    sos <- list.files(file.path(lib, 'libs'), pattern = '\\.so$', full.names = TRUE)
    for (so in sos) {
        # system2() without shell= needs no quoting; shQuote() would pass literal quotes
        # to otool and make it fail silently.
        link <- tryCatch(system2('otool', c('-L', so), stdout = TRUE, stderr = FALSE),
                         error = function(e) character(0))
        if (!length(link)) next
        lr <- grep('libR\\.dylib', link, value = TRUE)
        if (length(lr) && !grepl(r_home, lr[1], fixed = TRUE)) {
            foreign <- c(foreign, p); break
        }
    }
}
foreign <- unique(foreign)
if (length(foreign)) {
    cat('\n!! packages linked against a DIFFERENT R runtime than this one:\n')
    cat('     ', paste(foreign, collapse = ', '), '\n')
    cat('   They will segfault at dyn.load. Delete them and re-run this script:\n')
    cat('     Rscript -e \'remove.packages(c(',
        paste0('"', foreign, '"', collapse = ', '), '))\'', '\n\n', sep = '')
}

missing_conda <- conda_provided[!vapply(conda_provided, need, logical(1))]
if (length(missing_conda)) {
    stop('missing conda-provided packages: ', paste(missing_conda, collapse = ', '),
         '\nRecreate the env:  conda env create -f spatialpca_environment.yml',
         call. = FALSE)
}
cat('== compiled deps present (from conda):', paste(conda_provided, collapse = ', '), '\n')

missing_pure <- pure_r[!vapply(pure_r, need, logical(1))]
if (length(missing_pure)) {
    cat('== installing pure-R deps (no compilation):', paste(missing_pure, collapse = ', '), '\n')
    install.packages(missing_pure)
} else {
    cat('== pure-R deps already present\n')
}

spark_dir <- Sys.getenv('SP_SPARK_DIR')
spca_dir  <- Sys.getenv('SP_SPCA_DIR')

if (MINIMAL != '1' && !need('SPARK')) {
    cat('== installing SPARK from source (compiles Rcpp code; a few minutes)\n')
    install.packages(spark_dir, repos = NULL, type = 'source', dependencies = FALSE,
                     INSTALL_opts = '--no-test-load')
} else if (MINIMAL == '1') {
    cat('== skipping SPARK (not needed by 2d_scatter.ipynb; set SPATIALPCA_WITH_SPARK=1 to build it)\n')
} else {
    cat('== SPARK already present\n')
}

if (!need('SpatialPCA')) {
    cat('== installing SpatialPCA from source (no compilation needed)\n')
    install.packages(spca_dir, repos = NULL, type = 'source', dependencies = FALSE,
                     INSTALL_opts = '--no-test-load')
} else {
    cat('== SpatialPCA already present\n')
}

cat('\n--- versions ---\n')
for (p in c('SpatialPCA', 'SPARK', 'RSpectra', 'KernSmooth')) {
    v <- tryCatch(as.character(utils::packageVersion(p)), error = function(e) NA_character_)
    cat(sprintf('  %-12s %s\n', p, ifelse(is.na(v), 'not installed', v)))
}

# Actually load it: a successful install does not mean the namespace resolves.
ok <- tryCatch({ loadNamespace('SpatialPCA'); TRUE },
               error = function(e) { cat('  load error:', conditionMessage(e), '\n'); FALSE })
cat('\nSpatialPCA loadable:', ok, '\n')
RSCRIPT

# --- smoke test: run the notebook's script on synthetic data -------------------
echo
echo "== smoke-testing ERA5/notebooks/execute_spatialpca.R"
TMPDIR_SP="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_SP"' EXIT

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SP_SCRIPT="$REPO_ROOT/ERA5/notebooks/execute_spatialpca.R"

export SP_SMOKE_DIR="$TMPDIR_SP"
"$RSCRIPT" - <<'RSMOKE' >/dev/null
set.seed(42)
X <- matrix(rnorm(300 * 11), 300, 11)   # 300 locations x 11 ERA5 channels
C <- matrix(rnorm(300 * 3), 300, 3)     # normalized ECEF coords
d <- Sys.getenv('SP_SMOKE_DIR')
write.table(X, file.path(d, 'smoke_X.csv'), sep = ',', row.names = FALSE, col.names = FALSE)
write.table(C, file.path(d, 'smoke_coords.csv'), sep = ',', row.names = FALSE, col.names = FALSE)
RSMOKE

( cd "$TMPDIR_SP" && "$RSCRIPT" "$SP_SCRIPT" smoke 3 >/dev/null )

PYBIN=""
for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import numpy' >/dev/null 2>&1; then PYBIN="$c"; break; fi
done
if [[ -n "$PYBIN" ]]; then
    "$PYBIN" - "$TMPDIR_SP/res_smoke.csv" <<'PY'
import sys
import numpy as np
d = np.loadtxt(sys.argv[1], delimiter=",", skiprows=1)
assert d.ndim == 2 and d.shape[1] == 3, f"expected (N, 3), got {d.shape}"
print(f"  execute_spatialpca.R OK -> {d.shape}  (read exactly as the notebook does: np.loadtxt(..., skiprows=1))")
PY
else
    echo "  execute_spatialpca.R produced res_smoke.csv"
fi

echo
echo "Done. 2d_scatter.ipynb cell 'SpatialPCA' should now run."
