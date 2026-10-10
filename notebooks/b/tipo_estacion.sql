-- Tipo de estación (oficinas, residencial, recreativa, mixta) a partir de station_hour.
-- Regla y justificación: docs/reglas_operacion.md, "Umbrales para clasificar cada estación".
--
-- Usa solo el tramo de ENTRENAMIENTO: del 1 de julio al 31 de agosto de 2026.
-- Si el tramo cambia, se cambian las fechas de `params` y la lista de `festivos`.
-- Se corre con DuckDB desde la raíz del repo.
SET TimeZone = 'America/New_York';

WITH params AS (
    -- Sin -04 ni -05: con el SET TimeZone de arriba, DuckDB lee estas fechas en hora de Nueva York.
    SELECT TIMESTAMPTZ '2026-07-01 00:00:00' AS t0,   -- inicio del entrenamiento (incluido)
           TIMESTAMPTZ '2026-09-01 00:00:00' AS t1,   -- fin del entrenamiento (excluido)
           0.20  AS indice_corte,      -- |índice| >= 0.20: llegan (o salen) al menos 1.5 veces las que salen (o llegan)
           5     AS eventos_dia_min,   -- menos de 5 llegadas más salidas por día: no se clasifica (queda mixta)
           5     AS ventana_dia_min,   -- movimientos por día hábil exigidos de 7 a 10 Y de 16 a 19 para usar los índices
           20    AS recreativa_dia_min -- recreativa pide al menos 20 llegadas más salidas por día
),
-- Festivos del tramo: 4 de julio de 2026 cae en sábado y se observa el viernes 3.
festivos(fecha) AS (VALUES (DATE '2026-07-03')),
sh AS (
    SELECT s.station_id, s.departures, s.arrivals, s.departures_casual,
           hour(s.ts_hour) AS h,
           CAST(s.ts_hour AS DATE) AS fecha,
           CASE WHEN CAST(s.ts_hour AS DATE) IN (SELECT fecha FROM festivos) THEN 'festivo'
                WHEN dayofweek(s.ts_hour) IN (0, 6) THEN 'fin_de_semana'
                ELSE 'entre_semana' END AS tipo_dia
    FROM read_parquet('data/processed/station_hour.parquet') s, params p
    WHERE s.ts_hour >= p.t0 AND s.ts_hour < p.t1
),
dias AS (SELECT count(DISTINCT fecha) AS n,
                count(DISTINCT fecha) FILTER (WHERE tipo_dia = 'entre_semana') AS n_habiles FROM sh),
ind AS (
    SELECT station_id,
           sum(arrivals + departures) AS eventos,
           -- Mañana: entre semana de 7:00 a 10:00
           sum(arrivals - departures) FILTER (WHERE tipo_dia = 'entre_semana' AND h BETWEEN 7 AND 9)
             / nullif(sum(arrivals + departures) FILTER (WHERE tipo_dia = 'entre_semana' AND h BETWEEN 7 AND 9), 0)
             AS am_indice,
           sum(arrivals + departures) FILTER (WHERE tipo_dia = 'entre_semana' AND h BETWEEN 7 AND 9) AS am_mov,
           -- Tarde: entre semana de 16:00 a 19:00
           sum(arrivals - departures) FILTER (WHERE tipo_dia = 'entre_semana' AND h BETWEEN 16 AND 18)
             / nullif(sum(arrivals + departures) FILTER (WHERE tipo_dia = 'entre_semana' AND h BETWEEN 16 AND 18), 0)
             AS pm_indice,
           sum(arrivals + departures) FILTER (WHERE tipo_dia = 'entre_semana' AND h BETWEEN 16 AND 18) AS pm_mov,
           sum(arrivals + departures) FILTER (WHERE tipo_dia = 'fin_de_semana')
             / nullif(sum(arrivals + departures), 0) AS prop_fin_de_semana,
           sum(departures_casual) / nullif(sum(departures), 0) AS prop_casual
    FROM sh GROUP BY station_id
),
activas AS (
    SELECT ind.* FROM ind, dias, params p WHERE ind.eventos >= p.eventos_dia_min * dias.n
),
cortes AS (
    -- Percentiles entre estaciones activas (DuckDB pide el percentil escrito aquí, no en params):
    SELECT quantile_cont(prop_casual, 0.80) AS cas_q,          -- ocasionales para "recreativa"
           quantile_cont(prop_fin_de_semana, 0.75) AS fds_q    -- fin de semana para "recreativa"
    FROM activas
)
SELECT i.station_id, i.eventos, round(i.am_indice, 3) AS am_indice, round(i.pm_indice, 3) AS pm_indice,
       round(i.prop_fin_de_semana, 3) AS prop_fin_de_semana, round(i.prop_casual, 3) AS prop_casual,
       -- 1) Poca actividad: mixta. 2-3) Ir al trabajo: índice de la mañana fuerte, con suficientes
       -- movimientos para que no brinque, y que cambie de signo en la tarde (si gana o pierde todo
       -- el día suele ser por pendiente: mixta). 4) Ocio: sin patrón de ir al trabajo, con volumen,
       -- muchos ocasionales y mucho fin de semana. 5) El resto: mixta.
       CASE
         WHEN i.eventos < p.eventos_dia_min * d.n THEN 'mixta'
         WHEN i.am_mov >= p.ventana_dia_min * d.n_habiles AND i.pm_mov >= p.ventana_dia_min * d.n_habiles
              AND i.am_indice >= p.indice_corte AND i.pm_indice < 0 THEN 'oficinas'
         WHEN i.am_mov >= p.ventana_dia_min * d.n_habiles AND i.pm_mov >= p.ventana_dia_min * d.n_habiles
              AND i.am_indice <= -p.indice_corte AND i.pm_indice > 0 THEN 'residencial'
         WHEN (i.am_indice IS NULL OR abs(i.am_indice) < p.indice_corte)
              AND i.eventos >= p.recreativa_dia_min * d.n
              AND i.prop_casual >= c.cas_q AND i.prop_fin_de_semana >= c.fds_q THEN 'recreativa'
         ELSE 'mixta'
       END AS station_type,
       i.eventos < p.eventos_dia_min * d.n AS poca_actividad,
       round(c.cas_q, 3) AS corte_casual, round(c.fds_q, 3) AS corte_fds
FROM ind i, cortes c, params p, dias d
ORDER BY i.eventos DESC;
