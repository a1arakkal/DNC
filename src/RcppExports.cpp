// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// chol_inv
arma::mat chol_inv(arma::mat& x);
RcppExport SEXP _DNC_chol_inv(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(chol_inv(x));
    return rcpp_result_gen;
END_RCPP
}
// log_post_fun
double log_post_fun(arma::colvec& params, arma::mat& x, arma::colvec& y, arma::colvec& mu, arma::mat& sigma);
RcppExport SEXP _DNC_log_post_fun(SEXP paramsSEXP, SEXP xSEXP, SEXP ySEXP, SEXP muSEXP, SEXP sigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::colvec& >::type params(paramsSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::colvec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::colvec& >::type mu(muSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type sigma(sigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(log_post_fun(params, x, y, mu, sigma));
    return rcpp_result_gen;
END_RCPP
}
// log_post_fun_dnc
double log_post_fun_dnc(arma::colvec& params, arma::mat& x, arma::colvec& y, arma::colvec& mu, arma::mat& sigma, double& ratio);
RcppExport SEXP _DNC_log_post_fun_dnc(SEXP paramsSEXP, SEXP xSEXP, SEXP ySEXP, SEXP muSEXP, SEXP sigmaSEXP, SEXP ratioSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::colvec& >::type params(paramsSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::colvec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::colvec& >::type mu(muSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< double& >::type ratio(ratioSEXP);
    rcpp_result_gen = Rcpp::wrap(log_post_fun_dnc(params, x, y, mu, sigma, ratio));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_DNC_chol_inv", (DL_FUNC) &_DNC_chol_inv, 1},
    {"_DNC_log_post_fun", (DL_FUNC) &_DNC_log_post_fun, 5},
    {"_DNC_log_post_fun_dnc", (DL_FUNC) &_DNC_log_post_fun_dnc, 6},
    {NULL, NULL, 0}
};

RcppExport void R_init_DNC(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
