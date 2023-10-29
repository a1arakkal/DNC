#' chol_inv - a function to compute inverse of a symmetric matrxi
#' @return A matrix 
#' @details
#'    More summary needed here. 
#' 
#' @import Rcpp
#' @useDynLib DNC   
#' @export
chol_inv <- function(x) {
  .Call(`_DNC_chol_inv`, x, PACKAGE = "DNC")
}

#' log_post_fun - a function to compute the log posterior
#' @return A scalar 
#' @details
#'    More summary needed here. 
#' 
#' @import Rcpp
#' @useDynLib DNC   
#' @export
log_post_fun <- function(params, x, y, mu, sigma) {
  .Call('_DNC_log_post_fun', PACKAGE = 'DNC', params, x, y, mu, sigma)
}

#' log_post_fun_dnc - a function to compute the log posterior using power likelihood
#' @return A scalar 
#' @details
#'    More summary needed here. 
#' 
#' @import Rcpp
#' @useDynLib DNC   
#' @export
log_post_fun_dnc <- function(params, x, y, mu, sigma, ratio) {
  .Call('_DNC_log_post_fun_dnc', PACKAGE = 'DNC', params, x, y, mu, sigma, ratio)
}

