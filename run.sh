#!/bin/bash
# Script interactivo para ejecutar el pipeline principal del proyecto.
# Ejecutar desde la raiz: ./run.sh

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Seeking the Unicorn: analisis espectral${NC}"
echo -e "${BLUE}========================================${NC}\n"

if [ -d ".venv" ]; then
    VENV_DIR=".venv"
elif [ -d "venv" ]; then
    VENV_DIR="venv"
else
    echo -e "${YELLOW}No se encuentra el entorno virtual.${NC}"
    echo -e "${YELLOW}Creando entorno virtual en .venv ...${NC}\n"
    python3 -m venv .venv
    VENV_DIR=".venv"
    source "$VENV_DIR/bin/activate"
    echo -e "${GREEN}Instalando dependencias...${NC}"
    pip install -r requirements.txt
fi

source "$VENV_DIR/bin/activate"
echo -e "${GREEN}Entorno virtual activado: $VENV_DIR${NC}\n"

echo -e "Selecciona que script ejecutar:"
echo -e "  ${GREEN}1${NC}  - 00: Analisis espectral del grafo base"
echo -e "  ${GREEN}2${NC}  - 01: Analisis de una diada especifica"
echo -e "  ${GREEN}3${NC}  - 02: Analisis primario de diadas"
echo -e "  ${GREEN}4${NC}  - 03: Power iteration basico"
echo -e "  ${GREEN}5${NC}  - 03: Power iteration con Fiedler"
echo -e "  ${GREEN}6${NC}  - 04: Dinamica temporal"
echo -e "  ${GREEN}7${NC}  - 05: Contraejemplo P6xP8"
echo -e "  ${GREEN}8${NC}  - 06: Robustez de particiones"
echo -e "  ${GREEN}9${NC}  - 07: Reporte informacional + figura Andrade"
echo -e "  ${GREEN}10${NC} - 08: Prediccion temprana y transferencia"
echo -e "  ${GREEN}11${NC} - Pipeline principal (00, 01, 02, 04, 05, 06, 07, 08)"
echo -e "  ${GREEN}12${NC} - Solo activar entorno virtual"
echo -e ""
read -p "Opcion (1-12): " opcion

run_step() {
    local label="$1"
    local script="$2"
    echo -e "\n${YELLOW}${label}${NC}"
    python "$script"
}

case $opcion in
    1)
        run_step "Ejecutando 00_spectral_grid.py" "src/00_spectral_grid.py"
        ;;
    2)
        run_step "Ejecutando 01_single_dyad_analysis.py" "src/01_single_dyad_analysis.py"
        ;;
    3)
        run_step "Ejecutando 02_full_comparison.py" "src/02_full_comparison.py"
        ;;
    4)
        run_step "Ejecutando 03_power_iteration.py" "src/03_power_iteration.py"
        ;;
    5)
        run_step "Ejecutando 03_power_iteration_fiedler.py" "src/03_power_iteration_fiedler.py"
        ;;
    6)
        run_step "Ejecutando 04_temporal_dynamics.py" "src/04_temporal_dynamics.py"
        ;;
    7)
        run_step "Ejecutando 05_counterexample_P6xP8.py" "src/05_counterexample_P6xP8.py"
        ;;
    8)
        run_step "Ejecutando 06_partition_robustness.py" "src/06_partition_robustness.py"
        ;;
    9)
        run_step "Ejecutando 07_entropy_analysis.py" "src/07_entropy_analysis.py"
        ;;
    10)
        run_step "Ejecutando 08_early_prediction.py" "src/08_early_prediction.py"
        ;;
    11)
        run_step "[1/8] Grafo base" "src/00_spectral_grid.py"
        run_step "[2/8] Diada ejemplo" "src/01_single_dyad_analysis.py"
        run_step "[3/8] Analisis primario" "src/02_full_comparison.py"
        run_step "[4/8] Dinamica temporal" "src/04_temporal_dynamics.py"
        run_step "[5/8] Contraejemplo rectangular" "src/05_counterexample_P6xP8.py"
        run_step "[6/8] Robustez de particiones" "src/06_partition_robustness.py"
        run_step "[7/8] Entropia y figura resumen" "src/07_entropy_analysis.py"
        run_step "[8/8] Prediccion temprana y transferencia" "src/08_early_prediction.py"
        echo -e "\n${GREEN}Pipeline principal completado.${NC}"
        ;;
    12)
        echo -e "\n${GREEN}Entorno virtual activado. Usa 'deactivate' para salir.${NC}\n"
        exec $SHELL
        ;;
    *)
        echo -e "\n${YELLOW}Opcion invalida${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}Listo!${NC}"
