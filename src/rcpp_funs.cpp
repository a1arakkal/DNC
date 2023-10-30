
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::mat chol_inv(arma::mat& x){
  int n = x.n_rows;
  int p = x.n_cols;
  arma::mat L(n, p);
  
  for(int j = 0; j < p; j++){
    double sum = 0;
    for(int k = 0; k < j; k++){
      sum += L(j,k) * L(j,k);
    }
    L(j,j) = sqrt(x(j,j) - sum);
    
    for(int i = j + 1; i < p; i++){
      sum = 0;
      for (int k = 0; k < j; k++){
        sum += L(i,k) * L (j,k);
      }
      L(i,j) = (1.0 / L(j,j) * (x(i,j) - sum));
    }
  }
  arma::mat L_inv = inv(L);
  return L_inv.t()*L_inv;
}

// [[Rcpp::export]]
double log_post_fun(arma::colvec& params, arma::mat& x, arma::colvec& y, 
                    arma::colvec& mu, arma::mat& sigma){
  
  arma::colvec X_beta = x*params;
  arma::colvec p1 = arma::diagmat(y)*X_beta;
  arma::colvec p2 = arma::zeros(x.n_rows); 
  
  for(int j = 0; j < x.n_rows; j++){
    p2(j,0) = std::log(1 + std::exp(X_beta(j,0)));
  }
  arma::colvec diff = p1-p2;
  double p1_log_like = arma::accu(diff);

  arma::mat sigma_inv = chol_inv(sigma);
  arma::colvec diff_mean = params - mu;
  arma::colvec sigma_inv_diff_mean = sigma_inv*diff_mean;
  double p2_prior = 0.5*dot(diff_mean, sigma_inv_diff_mean);
  double out = p1_log_like - p2_prior;
  
  return out;
}  

// [[Rcpp::export]]
double log_post_fun_dnc(arma::colvec& params, arma::mat& x, arma::colvec& y, 
                        arma::colvec& mu, arma::mat& sigma, double& ratio){
  
  arma::colvec X_beta = x*params;
  arma::colvec p1 = arma::diagmat(y)*X_beta;
  arma::colvec p2 = arma::zeros(x.n_rows); 
  
  for(int j = 0; j < x.n_rows; j++){
    p2(j,0) = std::log(1 + std::exp(X_beta(j,0)));
  }
  arma::colvec diff = p1-p2;
  double p1_log_like = arma::accu(diff);

  arma::mat sigma_inv = chol_inv(sigma);
  arma::colvec diff_mean = params - mu;
  arma::colvec sigma_inv_diff_mean = sigma_inv*diff_mean;
  double p2_prior = 0.5*dot(diff_mean, sigma_inv_diff_mean);
  double out = (ratio*p1_log_like) - p2_prior;
  
  return out;
}  
  
// [[Rcpp::export]]

int MH(arma::mat& MH_draws, arma::mat& proposal_cov,
       arma::mat& x, arma::colvec& y, arma::colvec& mu, 
       arma::mat& sigma, double& ratio){
  int n = MH_draws.n_rows;
  arma::colvec beta_proposal = arma::zeros<arma::colvec>(MH_draws.n_cols);
  int acc_count = 0;
  
  // Obtain environment containing function
  Rcpp::Environment package_env("package:MASS"); 

  // Make function callable from C++
  Rcpp::Function mvtnomrfun = package_env["mvrnorm"]; 
  
  for(int j = 1; j < n; j++){
    MH_draws.row(j) = MH_draws.row(j-1);
    arma::colvec temp = MH_draws.row(j).t();
    beta_proposal = Rcpp::as<arma::colvec>(mvtnomrfun(1, temp, proposal_cov));
    double unif = R::runif(0,1);    
    double acc_prob = std::exp(log_post_fun_dnc(beta_proposal, x, y, mu, sigma, ratio) - 
                                 log_post_fun_dnc(temp, x, y, mu, sigma, ratio));
        
    if(unif < acc_prob){
      MH_draws.row(j) = beta_proposal.t();
      acc_count =  acc_count + 1;
    }
    
  }
    
  return acc_count;
}  