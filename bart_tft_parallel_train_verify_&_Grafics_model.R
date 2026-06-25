# ============================================================
# bart_tft_train_and_custom_verify_parallel_saved_model.R
# BART survival discreto para TFT
# - Entrena y guarda el modelo
# - Permite reutilizarlo
# - Soporta entrenamiento por múltiples cadenas en paralelo
# ============================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tibble)
  library(purrr)
  library(tidyr)
  library(BART)
  library(parallel)
  library(ggplot2)
  library(ggrepel)
})

# ------------------------------------------------------------
# 1) Configuración
# ------------------------------------------------------------
# Este código está hecho de modo que el nombre de los archivos
# tienen una misma raiz común de modo que cambiando tanto las
# variables de "version" y "version_lobby" como el nombre de
# los archivos con los dátos de los lobbies se puede ejecutar
# el código sin tener que realizar cambios en el resto del mismo

set.seed(123)


version = "full"
version_lobby = "full"

# Dirección en donde se accederá a los datos e importaran los resultados
setwd(paste0("C:/Documents/Resultados/BART_", version)) 

# Nombre del documento con los datos de entrenamiento
train_path  <- paste0("bart_training_dataset_expanded_", version, ".csv")

# Nombre del documento con los datos de validación
custom_path <- paste0("bart_tournament_lobbies_", version_lobby, ".csv")

# Nombre del modelo (como se guardará)
model_rds_path <- paste0("bart_tft_saved_", version, "_model_parallel.rds")

# Si existe modelo guardado y esto es FALSE, se reutiliza
force_retrain <- FALSE

# Hiperparámetros BART
ntree   <- 200
ndpost  <- 2000
nskip   <- 1000
time_grid_user <- NULL
rounds_of_interest <- c(18, 25, 32, 39, 40)

# ------------------------------------------------------------
# 2) Configuración de paralelismo
# ------------------------------------------------------------
# use_parallel = TRUE:
#   entrena varias cadenas MCMC en paralelo y luego combina draws.
# n_chains:
#   número de cadenas independientes.
# n_cores:
#   núcleos a utilizar.
#
# ------------------------------------------------------------
use_parallel <- TRUE
n_chains <- 4
n_cores  <- min(n_chains, max(1, parallel::detectCores() - 1))
base_seed <- 1234

# ------------------------------------------------------------
# 3) Lectura
# ------------------------------------------------------------
train_raw  <- readr::read_csv(train_path, show_col_types = FALSE)
custom_raw <- readr::read_csv(custom_path, show_col_types = FALSE)

# ------------------------------------------------------------
# 4) Estandarización de columnas
# ------------------------------------------------------------
standardize_input <- function(df) {
  df2 <- df
  
  if ("player_label" %in% names(df2)) {
    df2 <- df2 %>% rename(player_name = player_label)
  } else if ("player" %in% names(df2)) {
    df2 <- df2 %>% rename(player_name = player)
  } else {
    df2$player_name <- paste0("player_", seq_len(nrow(df2)))
  }
  
  drop_cols <- c("player_name", "riot_id", "puuid", "match_id", "game_datetime_utc")
  
  if (!all(c("status", "last_round") %in% names(df2))) {
    stop("El dataset debe contener al menos 'status' y 'last_round'.")
  }
  
  for (nm in names(df2)) {
    if (is.character(df2[[nm]]) &&
        nm %in% setdiff(names(df2), c("player_name", "riot_id", "puuid", "match_id", "game_datetime_utc"))) {
      suppressWarnings({
        tmp <- as.numeric(df2[[nm]])
      })
      if (!all(is.na(tmp))) df2[[nm]] <- tmp
    }
  }
  
  list(
    data = df2,
    drop_cols = intersect(drop_cols, names(df2))
  )
}

train_std  <- standardize_input(train_raw)
custom_std <- standardize_input(custom_raw)

train_df  <- train_std$data
custom_df <- custom_std$data

# ------------------------------------------------------------
# 5) Selección de covariables
# ------------------------------------------------------------
id_like_cols <- union(train_std$drop_cols, custom_std$drop_cols)

feature_cols <- setdiff(
  intersect(names(train_df), names(custom_df)),
  c(id_like_cols, "status", "last_round")
)

if (length(feature_cols) == 0) {
  stop("No se encontraron covariables comunes entre train y custom.")
}

message("Covariables utilizadas:")
print(feature_cols)

# ------------------------------------------------------------
# 6) Expansión discreta estilo Sparapani
# ------------------------------------------------------------
expand_survival_data <- function(data,
                                 id_col = "row_id",
                                 time_col = "last_round",
                                 status_col = "status",
                                 covariate_cols,
                                 time_grid = NULL) {
  stopifnot(all(c(id_col, time_col, status_col) %in% names(data)))
  
  if (is.null(time_grid)) {
    time_grid <- sort(unique(data[[time_col]][data[[status_col]] == 1]))
  }
  
  expanded_df <- purrr::map_dfr(seq_len(nrow(data)), function(i) {
    ti  <- data[[time_col]][i]
    di  <- as.integer(data[[status_col]][i])
    idi <- data[[id_col]][i]
    
    if (di == 1L) {
      grid_i <- time_grid[time_grid <= ti]
    } else {
      grid_i <- time_grid[time_grid < ti]
    }
    
    n_i <- length(grid_i)
    if (n_i == 0L) return(tibble())
    
    y_i <- if (di == 1L) c(rep(0L, n_i - 1L), 1L) else rep(0L, n_i)
    
    base_tbl <- tibble(
      row_id = rep(idi, n_i),
      time = grid_i,
      y = y_i,
      original_time = rep(ti, n_i),
      original_status = rep(di, n_i)
    )
    
    covars_i <- data[i, covariate_cols, drop = FALSE][rep(1, n_i), , drop = FALSE]
    bind_cols(base_tbl, covars_i)
  })
  
  list(expanded = expanded_df, time_grid = time_grid)
}

# ------------------------------------------------------------
# 7) Matriz de diseño
# ------------------------------------------------------------
build_design_matrix <- function(df, feature_cols) {
  rhs_terms <- c("time", feature_cols)
  x_formula <- reformulate(rhs_terms)
  mm <- model.matrix(x_formula, data = df)
  if ("(Intercept)" %in% colnames(mm)) {
    mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  }
  as.matrix(mm)
}

# ------------------------------------------------------------
# 8) Entrenamiento de una sola cadena
# ------------------------------------------------------------
train_single_chain <- function(x_train, y_train, ntree, ndpost_chain, nskip, seed_chain) {
  set.seed(seed_chain)
  fit <- BART::pbart(
    x.train = x_train,
    y.train = y_train,
    x.test  = x_train,
    ntree   = ntree,
    ndpost  = ndpost_chain,
    nskip   = nskip
  )
  fit
}

# ------------------------------------------------------------
# 9) Entrenamiento general (serial o paralelo por cadenas)
# ------------------------------------------------------------
train_model <- function(train_df, feature_cols,
                        ntree = 200, ndpost = 2000, nskip = 1000,
                        use_parallel = TRUE, n_chains = 4, n_cores = 4,
                        base_seed = 1234,
                        time_grid_user = NULL) {
  
  train_df <- train_df %>% mutate(row_id = row_number())
  
  if (!is.null(time_grid_user)) {
    time_grid_user <- sort(unique(as.integer(time_grid_user)))
    time_grid_user <- time_grid_user[time_grid_user >= 1]
  }
  
  exp_train <- expand_survival_data(
    data = train_df,
    id_col = "row_id",
    time_col = "last_round",
    status_col = "status",
    covariate_cols = feature_cols,
    time_grid = time_grid_user
  )
  
  x_train <- build_design_matrix(exp_train$expanded, feature_cols)
  y_train <- exp_train$expanded$y
  
  if (!use_parallel || n_chains <= 1) {
    message("Entrenamiento serial con una sola cadena...")
    fit <- train_single_chain(
      x_train = x_train,
      y_train = y_train,
      ntree = ntree,
      ndpost_chain = ndpost,
      nskip = nskip,
      seed_chain = base_seed
    )
    
    model_obj <- list(
      fits = list(fit),
      feature_cols = feature_cols,
      time_grid = exp_train$time_grid,
      x_train_colnames = colnames(x_train),
      train_params = list(
        ntree = ntree,
        ndpost_total = ndpost,
        ndpost_per_chain = ndpost,
        nskip = nskip,
        use_parallel = FALSE,
        n_chains = 1,
        n_cores = 1
      ),
      metadata = list(
        trained_at = as.character(Sys.time()),
        n_train_rows = nrow(train_df),
        n_expanded_rows = nrow(exp_train$expanded)
      )
    )
    
    return(model_obj)
  }
  
  ndpost_per_chain <- ceiling(ndpost / n_chains)
  seed_vec <- base_seed + seq_len(n_chains) - 1
  
  message("Entrenamiento paralelo por cadenas...")
  message("Cadenas: ", n_chains, " | Núcleos: ", n_cores,
          " | ndpost por cadena: ", ndpost_per_chain)
  
  if (.Platform$OS.type == "windows") {
    cl <- parallel::makeCluster(n_cores)
    on.exit(parallel::stopCluster(cl), add = TRUE)
    
    parallel::clusterExport(
      cl,
      varlist = c("x_train", "y_train", "ntree", "ndpost_per_chain", "nskip", "seed_vec"),
      envir = environment()
    )
    parallel::clusterEvalQ(cl, library(BART))
    
    fits <- parallel::parLapply(cl, seq_len(n_chains), function(i) {
      set.seed(seed_vec[i])
      BART::pbart(
        x.train = x_train,
        y.train = y_train,
        x.test  = x_train,
        ntree   = ntree,
        ndpost  = ndpost_per_chain,
        nskip   = nskip
      )
    })
  } else {
    fits <- parallel::mclapply(seq_len(n_chains), function(i) {
      train_single_chain(
        x_train = x_train,
        y_train = y_train,
        ntree = ntree,
        ndpost_chain = ndpost_per_chain,
        nskip = nskip,
        seed_chain = seed_vec[i]
      )
    }, mc.cores = n_cores)
  }
  
  model_obj <- list(
    fits = fits,
    feature_cols = feature_cols,
    time_grid = exp_train$time_grid,
    x_train_colnames = colnames(x_train),
    train_params = list(
      ntree = ntree,
      ndpost_total = ndpost_per_chain * n_chains,
      ndpost_per_chain = ndpost_per_chain,
      nskip = nskip,
      use_parallel = TRUE,
      n_chains = n_chains,
      n_cores = n_cores
    ),
    metadata = list(
      trained_at = as.character(Sys.time()),
      n_train_rows = nrow(train_df),
      n_expanded_rows = nrow(exp_train$expanded)
    )
  )
  
  model_obj
}

save_model_object <- function(model_obj, model_path) {
  saveRDS(model_obj, model_path)
  invisible(model_obj)
}

load_model_object <- function(model_path) {
  readRDS(model_path)
}

# ------------------------------------------------------------
# 10) Carga o entrenamiento
# ------------------------------------------------------------
if (file.exists(model_rds_path) && !isTRUE(force_retrain)) {
  message("Cargando modelo guardado desde: ", model_rds_path)
  bart_model <- load_model_object(model_rds_path)
  
  if (!setequal(bart_model$feature_cols, feature_cols)) {
    stop(
      "El modelo guardado fue entrenado con covariables distintas a las actuales. ",
      "Usa force_retrain <- TRUE para regenerarlo."
    )
  }
} else {
  message("Entrenando modelo BART...")
  bart_model <- train_model(
    train_df = train_df,
    feature_cols = feature_cols,
    ntree = ntree,
    ndpost = ndpost,
    nskip = nskip,
    use_parallel = use_parallel,
    n_chains = n_chains,
    n_cores = n_cores,
    base_seed = base_seed,
    time_grid_user = time_grid_user
  )
  save_model_object(bart_model, model_rds_path)
  message("Modelo guardado en: ", model_rds_path)
}

# ------------------------------------------------------------
# 11) Panel de predicción
# ------------------------------------------------------------
build_prediction_panel <- function(new_df, feature_cols, time_grid) {
  new_df <- new_df %>% mutate(row_id = row_number())
  
  purrr::map_dfr(seq_len(nrow(new_df)), function(i) {
    covars_i <- new_df[i, feature_cols, drop = FALSE][rep(1, length(time_grid)), , drop = FALSE]
    bind_cols(
      tibble(
        row_id = rep(new_df$row_id[i], length(time_grid)),
        time = time_grid
      ),
      covars_i
    )
  })
}

# ------------------------------------------------------------
# 12) Predicción desde modelo guardado
# ------------------------------------------------------------
# Combina draws de todas las cadenas. Cada cadena aporta su bloque
# posterior y se apilan verticalmente.
get_prob_draws_from_saved_model <- function(model_obj, x_test) {
  if (is.null(model_obj$fits) || length(model_obj$fits) == 0) {
    stop("El objeto cargado no contiene modelos entrenados.")
  }
  
  extract_prediction_matrix <- function(pred_obj, x_test) {
    # predict() devuelve objeto pbart (lista)
    if (inherits(pred_obj, "pbart")) {
      if (!is.null(pred_obj$prob.test)) {
        pred_try <- pred_obj$prob.test
      } else if (!is.null(pred_obj$yhat.test)) {
        pred_try <- pred_obj$yhat.test
      } else {
        stop("El objeto pbart no contiene 'prob.test' ni 'yhat.test'.")
      }
    } else if (is.list(pred_obj)) {
      if (!is.null(pred_obj$prob.test)) {
        pred_try <- pred_obj$prob.test
      } else if (!is.null(pred_obj$yhat.test)) {
        pred_try <- pred_obj$yhat.test
      } else {
        stop("La salida de predict() es una lista, pero no contiene 'prob.test' ni 'yhat.test'.")
      }
    } else {
      pred_try <- pred_obj
    }
    
    if (is.data.frame(pred_try)) {
      pred_try <- as.matrix(pred_try)
    }
    
    if (is.vector(pred_try)) {
      pred_try <- matrix(pred_try, nrow = 1)
    }
    
    nr <- nrow(pred_try)
    nc <- ncol(pred_try)
    ntest <- nrow(x_test)
    
    # Queremos filas = draws, columnas = observaciones
    if (!is.null(nr) && !is.null(nc) && nc != ntest && nr == ntest) {
      pred_try <- t(pred_try)
      nr <- nrow(pred_try)
      nc <- ncol(pred_try)
    }
    
    if (is.null(nc) || is.null(nr) || nc != ntest) {
      stop(
        "Dimensiones inesperadas en predict(): ",
        "nrow(pred_try)=", nr, ", ",
        "ncol(pred_try)=", nc, ", ",
        "nrow(x_test)=", ntest
      )
    }
    
    pred_try
  }
  
  draw_list <- lapply(seq_along(model_obj$fits), function(i) {
    fit_i <- model_obj$fits[[i]]
    pred_obj <- stats::predict(fit_i, newdata = x_test)
    extract_prediction_matrix(pred_obj, x_test)
  })
  
  do.call(rbind, draw_list)
}

predict_custom_survival <- function(model_obj, custom_df, rounds_of_interest = NULL,
                                    return_draws = TRUE) {
  # Compatibilidad con distintas versiones del objeto guardado
  feature_names <- if (!is.null(model_obj$feature_names)) {
    model_obj$feature_names
  } else if (!is.null(model_obj$feature_cols)) {
    model_obj$feature_cols
  } else {
    NULL
  }
  
  time_values <- if (!is.null(model_obj$time_values)) {
    model_obj$time_values
  } else if (!is.null(model_obj$time_grid)) {
    model_obj$time_grid
  } else {
    NULL
  }
  
  x_colnames <- if (!is.null(model_obj$x_colnames)) {
    model_obj$x_colnames
  } else if (!is.null(model_obj$x_train_colnames)) {
    model_obj$x_train_colnames
  } else {
    NULL
  }
  
  if (is.null(feature_names)) {
    stop("model_obj no contiene 'feature_names' ni 'feature_cols'.")
  }
  if (is.null(time_values)) {
    stop("model_obj no contiene 'time_values' ni 'time_grid'.")
  }
  if (is.null(x_colnames)) {
    stop("model_obj no contiene 'x_colnames' ni 'x_train_colnames'.")
  }
  
  time_values <- sort(unique(time_values))
  
  custom_base <- custom_df[, feature_names, drop = FALSE]
  
  # Crear expansión por tiempo
  custom_long <- do.call(
    rbind,
    lapply(seq_len(nrow(custom_base)), function(i) {
      row_i <- custom_base[rep(i, length(time_values)), , drop = FALSE]
      row_i$time <- time_values
      row_i$custom_row_id <- i
      row_i
    })
  )
  
  # Matriz de diseño
  x_test <- model.matrix(~ . - custom_row_id, data = custom_long)
  if ("(Intercept)" %in% colnames(x_test)) {
    x_test <- x_test[, colnames(x_test) != "(Intercept)", drop = FALSE]
  }
  
  # Alinear columnas con entrenamiento
  for (nm in setdiff(x_colnames, colnames(x_test))) {
    x_test <- cbind(x_test, 0)
    colnames(x_test)[ncol(x_test)] <- nm
  }
  x_test <- x_test[, x_colnames, drop = FALSE]
  
  # Draws de probabilidad discreta: filas = draws, columnas = observaciones
  prob_draws <- get_prob_draws_from_saved_model(model_obj, x_test)
  
  n_draws <- nrow(prob_draws)
  n_cases <- nrow(custom_base)
  n_times <- length(time_values)
  
  # Reorganizar a [draw, tiempo, caso]
  prob_array <- array(NA_real_, dim = c(n_draws, n_times, n_cases))
  
  idx <- 1
  for (case_i in seq_len(n_cases)) {
    cols_i <- idx:(idx + n_times - 1)
    prob_array[, , case_i] <- prob_draws[, cols_i, drop = FALSE]
    idx <- idx + n_times
  }
  
  # Supervivencia por draw y caso
  surv_array <- array(NA_real_, dim = c(n_draws, n_times, n_cases))
  for (case_i in seq_len(n_cases)) {
    surv_array[, , case_i] <- t(apply(prob_array[, , case_i, drop = FALSE][,,1], 1, function(p) cumprod(1 - p)))
  }
  
  # Resumen por tiempo y caso
  curves_list <- lapply(seq_len(n_cases), function(case_i) {
    tibble::tibble(
      custom_row_id = case_i,
      time = time_values,
      hazard_mean = colMeans(prob_array[, , case_i]),
      surv_mean = colMeans(surv_array[, , case_i]),
      surv_median = apply(surv_array[, , case_i], 2, median),
      surv_q025 = apply(surv_array[, , case_i], 2, quantile, probs = 0.025),
      surv_q975 = apply(surv_array[, , case_i], 2, quantile, probs = 0.975)
    )
  })
  
  curves_df <- dplyr::bind_rows(curves_list)
  
  custom_with_id <- custom_df
  if (!"custom_row_id" %in% names(custom_with_id)) {
    custom_with_id$custom_row_id <- seq_len(nrow(custom_with_id))
  }
  
  cols_to_attach <- setdiff(
    names(custom_with_id),
    c(feature_names, "custom_row_id")
  )
  
  attach_df <- custom_with_id[, c("custom_row_id", cols_to_attach), drop = FALSE]
  
  curves_df <- curves_df %>%
    dplyr::left_join(attach_df, by = "custom_row_id")
  
  rounds_of_interest <- sort(unique(rounds_of_interest))
  
  summary_df <- curves_df %>%
    dplyr::filter(time %in% rounds_of_interest) %>%
    dplyr::arrange(custom_row_id, time)
  
  out <- list(
    curves = curves_df,
    summary = summary_df
  )
  
  if (isTRUE(return_draws)) {
    prob_draws_long <- purrr::map_dfr(seq_len(n_cases), function(case_i) {
      as_tibble(prob_array[, , case_i]) %>%
        mutate(draw_id = row_number()) %>%
        tidyr::pivot_longer(
          cols = -draw_id,
          names_to = "time_index",
          values_to = "hazard_draw"
        ) %>%
        mutate(
          custom_row_id = case_i,
          time_index = as.integer(gsub("V", "", time_index)),
          time = time_values[time_index]
        ) %>%
        select(custom_row_id, draw_id, time, hazard_draw)
    })
    
    surv_draws_long <- purrr::map_dfr(seq_len(n_cases), function(case_i) {
      as_tibble(surv_array[, , case_i]) %>%
        mutate(draw_id = row_number()) %>%
        tidyr::pivot_longer(
          cols = -draw_id,
          names_to = "time_index",
          values_to = "surv_draw"
        ) %>%
        mutate(
          custom_row_id = case_i,
          time_index = as.integer(gsub("V", "", time_index)),
          time = time_values[time_index]
        ) %>%
        select(custom_row_id, draw_id, time, surv_draw)
    })
    
    prob_draws_long <- prob_draws_long %>%
      left_join(attach_df, by = "custom_row_id")
    
    surv_draws_long <- surv_draws_long %>%
      left_join(attach_df, by = "custom_row_id")
    
    out$prob_draws_long <- prob_draws_long
    out$surv_draws_long <- surv_draws_long
  }
  
  out
}

# ------------------------------------------------------------
# 13) Predicción custom
# ------------------------------------------------------------
custom_pred <- predict_custom_survival(
  model_obj = bart_model,
  custom_df = custom_df,
  rounds_of_interest = rounds_of_interest,
  return_draws = TRUE
)

# ------------------------------------------------------------
# 14) Exportación
# ------------------------------------------------------------
readr::write_csv(custom_pred$summary, paste0("bart_predictions_", version, "_summary.csv"))
readr::write_csv(custom_pred$curves,  paste0("bart_predictions_", version, "_curves.csv"))
readr::write_csv(custom_pred$prob_draws_long, paste0("bart_prob_draws_", version, "_long.csv"))
readr::write_csv(custom_pred$surv_draws_long, paste0("bart_surv_draws_", version, "_long.csv"))

message("Proceso finalizado.")
message("Archivos generados:")
message("- ", model_rds_path)
message(paste0("- bart_predictions_", version, "_summary.csv"))
message(paste0("- bart_predictions_", version, "_curves.csv"))
message(paste0("- bart_prob_draws_", version, "_long.csv"))
message(paste0("- bart_surv_draws_", version, "_long.csv"))
message("Resumen entrenamiento:")
print(bart_model$train_params)
print(bart_model$metadata)

#--------------------------------------------------------------------------------

#Gráfica - Curvas de supervivencia

curves_df <- read_csv(paste0("bart_predictions_", version, "_curves.csv"))
custom_df <- read_csv(paste0("bart_tournament_lobbies_", version_lobby, ".csv"))

match_col <- "match_id"
player_col <- "player_name"
player_col_custom <- "player"

plot_df <- curves_df %>%
  mutate(
    match_plot = .data[[match_col]],
    player_plot = .data[[player_col]]
  )

# Puntos reales: tomar, para cada jugador, la supervivencia predicha en su last_round real
real_points <- custom_df %>%
  mutate(
    match_plot = .data[[match_col]],
    player_plot = .data[[player_col_custom]]
  ) %>%
  select(match_plot, player_plot, last_round) %>%
  left_join(
    plot_df %>%
      select(match_plot, player_plot, time, surv_mean),
    by = c("match_plot", "player_plot", "last_round" = "time")
  )

matches <- unique(plot_df$match_plot)

pdf(paste0("survival_curves_by_match_", version, "_multipage.pdf"), width = 10, height = 6)

for (m in matches) {
  df_m <- plot_df %>% filter(match_plot == m)
  pts_m <- real_points %>% filter(match_plot == m)

  p_m <- ggplot(df_m, aes(x = time, y = surv_mean, color = player_plot, group = player_plot)) +
    geom_line(linewidth = 1) +
    geom_point(
      data = pts_m,
      aes(x = last_round, y = surv_mean, color = player_plot),
      inherit.aes = FALSE,
      size = 3,
      alpha = 0.9,
      show.legend = FALSE
    ) +
    labs(
      title = paste("Curvas de supervivencia -", m),
      x = "Ronda",
      y = "Probabilidad de seguir vivo",
      color = "Jugador"
    ) +
    theme_minimal()

  print(p_m)
}

dev.off()

#--------------------------------------------------------------------------------

#Gráfica - Intervalos de Confianza


# Ajusta estas rutas según tus archivos
surv_draws_path <- paste0("bart_surv_draws_", version, "_long.csv")

# Opcional. Déjalo en NULL si no quieres enriquecer con curvas medias.
curves_path <- paste0("bart_predictions_", version, "_curves.csv")

# Nombres de columnas de identificación
# Usa los nombres que existan realmente en tus archivos.
match_col  <- "match_id"
player_col <- "player_name"

# Rondas de interés para probabilidades de llegar vivo
rounds_of_interest <- c(18, 25, 32, 39, 40)

# Semilla para la simulación de ronda de eliminación por draw
set.seed(123)

# ------------------------------------------------------------
# 2) Utilidades
# ------------------------------------------------------------
pick_existing_col <- function(df, candidates, label) {
  hit <- intersect(candidates, names(df))
  if (length(hit) == 0) {
    stop("No se encontró una columna para ", label,
         ". Probadas: ", paste(candidates, collapse = ", "))
  }
  hit[[1]]
}

safe_first_crossing <- function(time, surv, threshold = 0.5, default_value = NA_real_) {
  idx <- which(surv <= threshold)
  if (length(idx) == 0) return(default_value)
  time[idx[1]]
}

# Convierte una curva media S(r) en PMF discreta de eliminación
# p(T=r) = S(r-1) - S(r), con S(r0-1)=1 para el primer tiempo.
curve_to_elim_summary <- function(df_player, rounds_of_interest = NULL) {
  df_player <- df_player %>% arrange(time)
  
  time_vals <- df_player$time
  surv_mean <- df_player$surv_mean
  
  surv_prev <- c(1, head(surv_mean, -1))
  p_elim <- pmax(0, surv_prev - surv_mean)
  
  if (sum(p_elim, na.rm = TRUE) > 0) {
    p_elim <- p_elim / sum(p_elim, na.rm = TRUE)
  }
  
  expected_elim_round <- sum(time_vals * p_elim, na.rm = TRUE)
  median_elim_round <- safe_first_crossing(time_vals, surv_mean, 0.5, default_value = max(time_vals, na.rm = TRUE))
  modal_elim_round <- time_vals[which.max(p_elim)][1]
  
  out <- tibble(
    expected_elim_round = expected_elim_round,
    median_elim_round_curve = median_elim_round,
    modal_elim_round = modal_elim_round
  )
  
  if (all(c("surv_q025", "surv_q975") %in% names(df_player))) {
    lower_approx <- safe_first_crossing(time_vals, df_player$surv_q975, 0.5, default_value = min(time_vals, na.rm = TRUE))
    upper_approx <- safe_first_crossing(time_vals, df_player$surv_q025, 0.5, default_value = max(time_vals, na.rm = TRUE))
    
    out <- out %>%
      mutate(
        elim_round_lower_approx = lower_approx,
        elim_round_upper_approx = upper_approx
      )
  }
  
  if (!is.null(rounds_of_interest)) {
    probs_alive <- map_dbl(rounds_of_interest, function(r) {
      idx <- which(df_player$time == r)
      if (length(idx) == 0) return(NA_real_)
      df_player$surv_mean[idx[1]]
    })
    names(probs_alive) <- paste0("prob_alive_round_", rounds_of_interest)
    
    out <- bind_cols(out, as_tibble_row(as.list(probs_alive)))
  }
  
  out
}

# A partir de draws de supervivencia:
# 1) construye p_elim por draw
# 2) simula una ronda de eliminación por draw
# 3) resume cuantiles
compute_elimination_intervals_from_draws <- function(surv_draws_long, match_col, player_col) {
  required_cols <- c("custom_row_id", "draw_id", "time", "surv_draw", match_col, player_col)
  missing_cols <- setdiff(required_cols, names(surv_draws_long))
  if (length(missing_cols) > 0) {
    stop("Faltan columnas en surv_draws_long: ", paste(missing_cols, collapse = ", "))
  }
  
  elim_draws <- surv_draws_long %>%
    arrange(custom_row_id, draw_id, time) %>%
    group_by(custom_row_id, draw_id) %>%
    mutate(
      surv_prev = lag(surv_draw, default = 1),
      p_elim = pmax(0, surv_prev - surv_draw)
    ) %>%
    ungroup() %>%
    group_by(custom_row_id, draw_id) %>%
    mutate(
      p_sum = sum(p_elim, na.rm = TRUE),
      p_elim = ifelse(p_sum > 0, p_elim / p_sum, 0)
    ) %>%
    summarise(
      elim_round_draw = sample(time, size = 1, prob = p_elim),
      .groups = "drop"
    ) %>%
    left_join(
      surv_draws_long %>%
        distinct(custom_row_id, .data[[match_col]], .data[[player_col]]),
      by = "custom_row_id"
    )
  
  elim_intervals <- elim_draws %>%
    group_by(custom_row_id, .data[[match_col]], .data[[player_col]]) %>%
    summarise(
      n_draws = n(),
      elim_round_mean = mean(elim_round_draw, na.rm = TRUE),
      elim_round_sd = sd(elim_round_draw, na.rm = TRUE),
      elim_round_median = median(elim_round_draw, na.rm = TRUE),
      elim_round_q025 = quantile(elim_round_draw, 0.025, na.rm = TRUE),
      elim_round_q05 = quantile(elim_round_draw, 0.05, na.rm = TRUE),
      elim_round_q10 = quantile(elim_round_draw, 0.10, na.rm = TRUE),
      elim_round_q25 = quantile(elim_round_draw, 0.25, na.rm = TRUE),
      elim_round_q75 = quantile(elim_round_draw, 0.75, na.rm = TRUE),
      elim_round_q90 = quantile(elim_round_draw, 0.90, na.rm = TRUE),
      elim_round_q95 = quantile(elim_round_draw, 0.95, na.rm = TRUE),
      elim_round_q975 = quantile(elim_round_draw, 0.975, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(.data[[match_col]], desc(elim_round_mean))
  
  list(
    elim_draws = elim_draws,
    elim_intervals = elim_intervals
  )
}

# Rankings por lobby a partir de estadísticos de eliminación
build_lobby_rankings <- function(elim_stats, match_col, player_col) {
  elim_stats %>%
    group_by(.data[[match_col]]) %>%
    arrange(desc(elim_round_mean), .by_group = TRUE) %>%
    mutate(
      rank_by_mean_elim = row_number(desc(elim_round_mean)),
      rank_by_median_elim = row_number(desc(elim_round_median))
    ) %>%
    ungroup() %>%
    select(.data[[match_col]], .data[[player_col]], everything())
}

# ------------------------------------------------------------
# 3) Lectura de draws
# ------------------------------------------------------------
surv_draws_long <- readr::read_csv(surv_draws_path, show_col_types = FALSE)

# Resolver nombres reales si el archivo usa variantes
match_col_resolved <- pick_existing_col(surv_draws_long, c(match_col, "match_id", "lobby_id"), "match")
player_col_resolved <- pick_existing_col(surv_draws_long, c(player_col, "player_name", "player_label", "player"), "player")

# ------------------------------------------------------------
# 4) Intervalos predictivos a partir de draws
# ------------------------------------------------------------
elim_obj <- compute_elimination_intervals_from_draws(
  surv_draws_long = surv_draws_long,
  match_col = match_col_resolved,
  player_col = player_col_resolved
)

elim_draws <- elim_obj$elim_draws
elim_intervals <- elim_obj$elim_intervals

readr::write_csv(elim_intervals, paste0("bart_elimination_intervals_", version, ".csv"))
readr::write_csv(elim_draws, paste0("bart_elimination_draws_", version, ".csv"))

# ------------------------------------------------------------
# 5) Enriquecimiento opcional con curvas medias
# ------------------------------------------------------------
if (!is.null(curves_path) && file.exists(curves_path)) {
  curves_df <- readr::read_csv(curves_path, show_col_types = FALSE)
  
  match_col_curves <- pick_existing_col(curves_df, c(match_col_resolved, "match_id", "lobby_id"), "match en curvas")
  player_col_curves <- pick_existing_col(curves_df, c(player_col_resolved, "player_name", "player_label", "player"), "player en curvas")
  
  if (!"surv_mean" %in% names(curves_df)) {
    stop("El archivo de curvas no contiene 'surv_mean'.")
  }
  
  curves_stats <- curves_df %>%
    group_by(custom_row_id, .data[[match_col_curves]], .data[[player_col_curves]]) %>%
    group_modify(~ curve_to_elim_summary(.x, rounds_of_interest = rounds_of_interest)) %>%
    ungroup()
  
  # Renombrar por consistencia si los nombres resueltos difieren
  if (match_col_curves != match_col_resolved) {
    curves_stats <- curves_stats %>% rename(!!match_col_resolved := .data[[match_col_curves]])
  }
  if (player_col_curves != player_col_resolved) {
    curves_stats <- curves_stats %>% rename(!!player_col_resolved := .data[[player_col_curves]])
  }
  
  elim_stats <- elim_intervals %>%
    left_join(
      curves_stats,
      by = c("custom_row_id", match_col_resolved, player_col_resolved)
    ) %>%
    arrange(.data[[match_col_resolved]], desc(elim_round_mean))
  
  lobby_rankings <- build_lobby_rankings(
    elim_stats = elim_stats,
    match_col = match_col_resolved,
    player_col = player_col_resolved
  )
  
  readr::write_csv(elim_stats, paste0("bart_elimination_stats_", version, ".csv"))
  readr::write_csv(lobby_rankings, paste0("bart_elimination_rankings_", version, ".csv"))
  
  message("Archivos generados:")
  message(paste0("- bart_elimination_intervals_", version, ".csv"))
  message(paste0("- bart_elimination_draws_", version, ".csv"))
  message(paste0("- bart_elimination_stats_", version, ".csv"))
  message(paste0("- bart_elimination_rankings_", version, ".csv"))
} else {
  message("Archivo de curvas no provisto o no encontrado. Solo se generaron:")
  message(paste0("- bart_elimination_intervals_", version, ".csv"))
  message(paste0("- bart_elimination_draws_", version, ".csv"))
}

message("Proceso finalizado.")



#--------------------------------------------------------------------------------

intervals_df <- read_csv(paste0("bart_elimination_intervals_", version, ".csv"))

real_player_df <- custom_df %>%
  transmute(
    match_id,
    player_name = player,
    real_last_round = last_round
  ) %>%
  distinct()

# real_lobby_df <- custom_df %>%
#   group_by(match_id) %>%
#   summarise(
#     real_lobby_end_round = max(last_round, na.rm = TRUE),
#     .groups = "drop"
#   )

plot_df <- intervals_df %>%
  left_join(real_player_df, by = c("match_id", "player_name")) # %>%
#  left_join(real_lobby_df, by = "match_id")

matches <- unique(plot_df$match_id)

pdf(paste0("bart_elimination_intervals_by_match_90_95_", version, ".pdf"), width = 11, height = 6.5)

for (m in matches) {
  df_m <- plot_df %>%
    filter(match_id == m) %>%
    arrange(elim_round_median) %>%
    mutate(player_name = factor(player_name, levels = player_name))
  
  p <- ggplot(df_m, aes(y = player_name)) +
    # Banda 95% (más clara, más larga)
    geom_segment(
      aes(
        x = elim_round_q025, xend = elim_round_q975,
        y = player_name, yend = player_name
      ),
      linewidth = 3.2,
      color = "lightskyblue2",
      alpha = 0.55
    ) +
    # Banda 90% (más oscura, por encima)
    geom_segment(
      aes(
        x = elim_round_q05, xend = elim_round_q95,
        y = player_name, yend = player_name
      ),
      linewidth = 5,
      color = "steelblue3",
      alpha = 0.9
    ) +
    # Mediana predictiva
    geom_point(
      aes(x = elim_round_median),
      color = "navy",
      size = 3
    ) +
    # Ronda real del jugador
    geom_point(
      aes(x = real_last_round),
      color = "firebrick",
      size = 3.2,
      shape = 17
    ) +
    # Fin real del lobby
    # geom_vline(
    #   aes(xintercept = real_lobby_end_round),
    #   linetype = "dashed",
    #   linewidth = 0.8,
    #   color = "black"
    # ) +
    scale_x_continuous(
      limits = c(18, 41),
      breaks = seq(18, 41, by = 1)
    ) +
    labs(
      title = paste("Intervalos predictivos de eliminación (90% y 95%) -", m),
      subtitle = "Azul oscuro: 90% | Azul claro: 95% | Punto azul: mediana predicha | Triángulo rojo: ronda real del jugador",
      x = "Ronda",
      y = "Jugador"
    ) +
    theme_minimal() +
    theme(
      panel.grid.minor = element_blank()
    )
  
  print(p)
}

dev.off()


















