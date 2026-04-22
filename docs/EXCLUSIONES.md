# EXCLUSIONES DEL ANALISIS PRIMARIO

## Regla del set primario

Dataset primario:
- `data/raw/humans_only_absent.csv`

Ventana usada en el analisis primario:
- rondas `absent` con `Round >= 40`

Una diada entra al set primario solo si:
- tiene exactamente 2 jugadores en esa ventana,
- induce una particion dura `S_obs = {v : f1(v) > f2(v)}`,
- el tamano de `S_obs` cae entre 10 y 54 nodos,
- y `h(S_obs)` es finita y distinta de cero.

En la practica, todas las exclusiones del analisis primario actual provienen del tamano de la particion.

## Flujo de muestra

- Diadas auditadas: 45
- Diadas incluidas en el set primario: 29
- Excluidas por particion muy pequena: 8
- Excluidas por particion muy grande: 8

## Diadas excluidas

### Particion muy pequena

Estas diadas producen `|S_obs| < 10` en la ventana `Round >= 40`:

- `216-713`
- `261-970`
- `313-199`
- `356-137`
- `379-897`
- `462-640`
- `475-186`
- `880-349`

### Particion muy grande

Estas diadas producen `|S_obs| > 54` en la ventana `Round >= 40`:

- `352-425`
- `359-904`
- `416-710`
- `483-710`
- `487-811`
- `590-286`
- `636-625`
- `938-219`

## Lectura critica

- Las exclusiones no son ruido arbitrario: muchas de estas diadas tienen categorias tardias `ALL`, `NOTHING` o `RS`.
- Eso significa que el filtro de conductancia esta seleccionando sobre todo las diadas con geometria de biparticion axial razonable y dejando fuera estrategias que el paper original si trata como focales o al menos relevantes.
- Por eso el analisis espectral actual debe describirse como **parcial**:
  - fuerte para `LR/TB`,
  - debil o directamente inapropiado para `ALL/NOTHING`,
  - y no interpretable como explicacion total del fenomeno de SODCL.

## Distribucion cualitativa de excluidas

Modo de categoria tardia entre las 16 excluidas:

- `ALL`: 7
- `RS`: 6
- `NOTHING`: 3

Esta distribucion refuerza que la exclusion no es solo tecnica. Tambien delimita el tipo de organizacion espacial que la metrica de conductancia puede capturar limpiamente.
