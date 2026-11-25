# usage: source env.sh   (after ./setup.sh)
VKS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$VKS_DIR/.venv}"
LIBERO_ROOT="${LIBERO_ROOT:-$VKS_DIR/third_party/LIBERO-PRO}"
source "$VENV/bin/activate"
export PYTHONPATH="$VKS_DIR"
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl          # headless rendering; use osmesa if EGL is not available

# LIBERO asks for its paths interactively on first import; write the config once instead
export LIBERO_CONFIG_PATH="$VKS_DIR/.libero"
if [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]; then
    mkdir -p "$LIBERO_CONFIG_PATH"
    L="$LIBERO_ROOT/libero/libero"
    printf 'benchmark_root: %s\nbddl_files: %s/bddl_files\ninit_states: %s/init_files\ndatasets: %s/datasets\nassets: %s/assets\n' \
        "$L" "$L" "$L" "$VKS_DIR" "$L" > "$LIBERO_CONFIG_PATH/config.yaml"
fi
