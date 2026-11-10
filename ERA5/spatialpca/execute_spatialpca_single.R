args <- commandArgs(trailingOnly = TRUE)
id <- args[1]
n_components <- if (length(args) >= 2) as.integer(args[2]) else 10

suppressPackageStartupMessages(library(SpatialPCA))

cat("reading in data\n")
X <- as.matrix(read.csv(paste0(id, "_X.csv"), header = FALSE))
coords <- as.matrix(read.csv(paste0(id, "_coords.csv"), header = FALSE))

# SpatialPCA expects genes/features x locations; skip CreateSpatialPCAObject's
# Seurat/SCTransform normalization (built for non-negative RNA-seq counts) and set
# normalized_expr directly, since our features are already normalized upstream.
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

cat("building kernel\n")
object <- SpatialPCA_buildKernel(object, kerneltype = "gaussian", bandwidthtype = "SJ")

cat("estimating loading\n")
object <- SpatialPCA_EstimateLoading(object, fast = TRUE, SpatialPCnum = n_components)

cat("computing spatial PCs\n")
object <- SpatialPCA_SpatialPCs(object, fast = TRUE)

out_file <- paste0("res_", id, ".csv")
write.csv(t(object@SpatialPCs), out_file, row.names = FALSE)
cat("file saved as ", out_file, "\n", sep = "")
