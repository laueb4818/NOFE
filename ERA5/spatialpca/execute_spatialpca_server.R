suppressPackageStartupMessages(library(SpatialPCA))

# Persistent server: load the package once, then serve requests read one-per-line from
# stdin as "<id> <n_components>". Each request reads {id}_X.csv/{id}_coords.csv, computes
# the SpatialPCA embedding, writes res_{id}.csv, and replies with one line ("OK" or
# "ERROR: <message>") on stdout so the calling process knows when the result is ready.
# Avoids paying R startup + library(SpatialPCA) load cost (Seurat, SPARK, etc.) on every
# single call, which dominates runtime when called many times in a loop.

embed_one <- function(id, n_components) {
    X <- as.matrix(read.csv(paste0(id, "_X.csv"), header = FALSE))
    coords <- as.matrix(read.csv(paste0(id, "_coords.csv"), header = FALSE))

    expr <- t(X)
    colnames(expr) <- paste0("loc", seq_len(ncol(expr)))
    rownames(coords) <- colnames(expr)

    object <- new(
        "SpatialPCA",
        counts = expr,
        normalized_expr = expr,
        location = coords,
        project = id
    )
    object <- SpatialPCA_buildKernel(object, kerneltype = "gaussian", bandwidthtype = "SJ")
    object <- SpatialPCA_EstimateLoading(object, fast = TRUE, SpatialPCnum = n_components)
    object <- SpatialPCA_SpatialPCs(object, fast = TRUE)

    write.csv(t(object@SpatialPCs), paste0("res_", id, ".csv"), row.names = FALSE)
}

con <- file("stdin", open = "r")
cat("READY\n")
flush(stdout())

while (TRUE) {
    line <- readLines(con, n = 1)
    if (length(line) == 0 || line == "QUIT") break

    parts <- strsplit(line, " ")[[1]]
    id <- parts[1]
    n_components <- if (length(parts) >= 2) as.integer(parts[2]) else 10

    result <- tryCatch({
        # SpatialPCA_buildKernel/EstimateLoading/SpatialPCs print progress messages via
        # cat()/print() -- silence them so they don't corrupt the one-line-per-request
        # protocol on stdout.
        sink(nullfile())
        embed_one(id, n_components)
        sink()
        "OK"
    }, error = function(e) {
        if (sink.number() > 0) sink()
        paste("ERROR:", gsub("\n", " ", conditionMessage(e)))
    })

    cat(result, "\n")
    flush(stdout())
}
