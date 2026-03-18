#!/bin/bash
# Script de ejecucion para el proyecto de Analisis Espectral
# Automatiza la activacion del entorno virtual y ejecucion de scripts
# Ejecutar desde la raiz del proyecto: ./run.sh

# Colores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Analisis Espectral de Division del Trabajo${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Detectar entorno virtual disponible (.venv tiene prioridad sobre venv)
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

# Menu de opciones
echo -e "Selecciona que script ejecutar:"
echo -e "  ${GREEN}1${NC} - Analisis espectral del grafo base         (src/00_spectral_grid.py)"
echo -e "  ${GREEN}2${NC} - Analisis de una diada especifica          (src/01_single_dyad_analysis.py)"
echo -e "  ${GREEN}3${NC} - Comparacion completa de todas las diadas  (src/02_full_comparison.py)"
echo -e "  ${GREEN}4${NC} - Power iteration basico                    (src/03_power_iteration.py)"
echo -e "  ${GREEN}5${NC} - Power iteration con Fiedler               (src/03_power_iteration_fiedler.py)"
echo -e "  ${GREEN}6${NC} - Ejecutar pipeline completo (scripts 00-02 en secuencia)"
echo -e "  ${GREEN}7${NC} - Solo activar entorno virtual (shell interactivo)"
echo -e ""
read -p "Opcion (1-7): " opcion

case $opcion in
    1)
        echo -e "\n${BLUE}Ejecutando analisis espectral del grafo...${NC}\n"
        python src/00_spectral_grid.py
        ;;
    2)
        echo -e "\n${BLUE}Ejecutando analisis de diada individual...${NC}\n"
        python src/01_single_dyad_analysis.py
        ;;
    3)
        echo -e "\n${BLUE}Ejecutando comparacion completa...${NC}\n"
        python src/02_full_comparison.py
        ;;
    4)
        echo -e "\n${BLUE}Ejecutando power iteration basico...${NC}\n"
        python src/03_power_iteration.py
        ;;
    5)
        echo -e "\n${BLUE}Ejecutando power iteration con Fiedler...${NC}\n"
        python src/03_power_iteration_fiedler.py
        ;;
    6)
        echo -e "\n${BLUE}Ejecutando pipeline completo...${NC}\n"
        echo -e "${YELLOW}[1/3] Analisis espectral del grafo...${NC}"
        python src/00_spectral_grid.py
        echo -e "\n${YELLOW}[2/3] Analisis de diada individual...${NC}"
        python src/01_single_dyad_analysis.py
        echo -e "\n${YELLOW}[3/3] Comparacion completa...${NC}"
        python src/02_full_comparison.py
        echo -e "\n${GREEN}Pipeline completo ejecutado!${NC}"
        ;;
    7)
        echo -e "\n${GREEN}Entorno virtual activado. Usa 'deactivate' para salir.${NC}\n"
        exec $SHELL
        ;;
    *)
        echo -e "\n${YELLOW}Opcion invalida${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}Listo!${NC}"
