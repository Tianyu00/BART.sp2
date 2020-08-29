#include <RcppArmadillo.h>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/version.hpp>
#include <cmath>
#include <iostream>

using namespace Rcpp;

const double log2pi = std::log(2.0 * M_PI);









// https://stackoverflow.com/questions/47274696/segment-fault-when-using-rcpp-armadillo-and-openmp-prarallel-with-user-defined-f
arma::mat MaternFun(arma::mat distmat, arma::vec covparms) {

   std::cout << "Boost version: " 
        << BOOST_VERSION / 100000
        << "."
        << BOOST_VERSION / 100 % 1000
        << "."
        << BOOST_VERSION % 100 
        << std::endl;
  // covparms: 3 parameters: tau^2, range, smoothness

  double covparms0 = covparms(0);
  double covparms1 = exp(covparms(1));
  double covparms2 = exp(covparms(2));
  if (covparms2>10) {covparms(2)=10;covparms2=10;}

  int d1 = distmat.n_rows;
  int d2 = distmat.n_cols;
  int j1;
  int j2;
  arma::mat covmat(d1,d2);
  double scaledist;

  double normcon = 1 /
    (pow(2.0, covparms2 - 1)* boost::math::tgamma(covparms2));

  for (j1 = 0; j1 < d1; ++j1){
    for (j2 = 0; j2 < d2; ++j2){
      if ( distmat(j1, j2) == 0 ){
        covmat(j1, j2) = 1;
      } else {
        scaledist = distmat(j1, j2)/covparms1;
        covmat(j1, j2) = normcon * pow( scaledist, covparms2 ) *
          boost::math::cyl_bessel_k(covparms2, scaledist);
      }
    }
  }
  if(covmat.has_nan()){covmat.print();covparms.print();}
  if(covmat.has_inf()){covmat.print();covparms.print();}
  covmat = covmat * covparms0;

  return covmat;
}





// https://stackoverflow.com/questions/41884478/why-cant-i-get-the-square-root-of-this-symmetric-positive-definite-matrix-in-ar
arma::mat raiz(const arma::mat A){
  arma::vec D;
  arma::mat B;
  arma::eig_sym(D,B,A);

  unsigned int n = D.n_elem;

  arma::mat G(n,n,arma::fill::zeros);

  for(unsigned int i=0;i<n;i++){
      if(D(i)<0 ) D(i)=0;
      G(i,i)=sqrt(D(i));
  }

  return B*G*B.t();
}



// https://gallery.rcpp.org/articles/simulate-multivariate-normal/
//arma::mat mvrnormArma(int n, arma::vec mu, arma::mat sigma) {
//   int ncols = sigma.n_cols;
//   arma::mat Y = arma::randn(n, ncols);
//   arma::vec d(ncols);
//   arma::mat Q(ncols,ncols);
//   sigma = (sigma+sigma.t())/2;
//   arma::eig_sym(d, Q, sigma);
//   cout << ncols << ' ' << d[0] << ' ' << d[ncols-1] << ' ';
//   //cout << endl;
//
//   double Eps = 1e-9;
//   for(size_t iii=0; iii<ncols;iii++) {
//     if(d(iii) < Eps){d(iii) = Eps;}}
//   arma::mat dd = arma::diagmat(d);
//   sigma = Q * dd * Q.t();
//
//   return arma::repmat(mu, 1, n).t() + Y * arma::chol(sigma);
//}
arma::mat mvrnormArma(int n, arma::vec mu, arma::mat sigma) {
   int ncols = sigma.n_cols;
   arma::mat Y = arma::randn(n, ncols);
   arma::vec d(ncols);
   arma::mat Q(ncols,ncols);
   arma::eig_sym(d, Q, sigma);
   //cout << ncols << ' ' << d[0] << ' ' << d[ncols-1] << ' ';
//   sigma = (sigma + sigma.t())/2;
   return arma::repmat(mu, 1, n).t() + Y * arma::chol(sigma);
}





arma::vec Mahalanobis(arma::mat x, arma::rowvec center, arma::mat cov) {
    int n = x.n_rows;
    arma::mat x_cen;
    x_cen.copy_size(x);
    for (int i=0; i < n; i++) {
        x_cen.row(i) = x.row(i) - center;
    }
    return sum((x_cen * cov.i()) % x_cen, 1);
}


arma::vec dmvnrm_arma(arma::mat x, arma::rowvec mean, arma::mat sigma, bool log = true) {
    arma::vec distval = Mahalanobis(x,  mean, sigma);
    arma::mat tt = arma::symmatu(sigma);
    if(!tt.is_symmetric()) {printf("dmvnrm\n");tt.print();}
    double logdet = sum(arma::log(arma::eig_sym(tt)));
    arma::vec logretval = -( (x.n_cols * log2pi + logdet + distval)/2  ) ;

    if (log) {
        return(logretval);
    } else {
        return(exp(logretval));
    }
}

