function [agx_ECEF,agy_ECEF,agz_ECEF]=grav_odp(x,y,z,mu,RE,C,S)

%  This function computes the gravitational accelerations using a
%  spherical harmonic expansion of the geopotential.  The geopotential
%  is expressed in ECEF coordinates, thus the input coordinates need to
%  be in ECEF.  The acceleration must be expressed in inertial
%  coordinates for ease of integration. The gravity coefficients are
%  assumed to already be normalized.
%
%  Reference:  Pines, Samuel, "Uniform Representation of the Gravitational
%              Potential and its Derivatives," AIAA Journal, Vol. 11, 
%              No. 11, November, 1973, pp. 1508-1511.
%
%  INPUTS:
%     x,y,z     Spacecraft coordinates in ECI coordinates
%     mu        Earth's Gravitational Constant
%     RE        Earth's Mean Equitorial Radius
%     C(i,j)    A matrix containing the C coefficients,
%               which are used in the spherical harmonic
%               expansion of the geopotential
%     S(i,j)    A matrix containing the S coefficients,
%               which are similar to C coefficients.
%  OUTPUTS:
%     agx_ECEF   x-component (ECEF) of the acceleration vector
%     agy_ECEF   y-component (ECEF) of the acceleration vector
%     agz_ECEF   z-component (ECEF) of the acceleration vector
%
% Initialize a few variables and determine their size

[nmaxp1,mmaxp1] = size(C);
nmax = nmaxp1-1;
mmax = mmaxp1-1;
Anm = zeros(nmaxp1+1,1);
Anm1 = zeros(nmaxp1+1,1);
Anm2 = zeros(nmaxp1+2,1);
R = zeros(nmaxp1+1,1);
I = zeros(nmaxp1+1,1);
rb2 = x*x + y*y + z*z;
rb = sqrt(rb2);
mur2 = mu/rb2;
mur3 = mur2/rb;

% direction of spacecraft position
s = x/rb;
t = y/rb;
u = z/rb;

%	/* ********************************************************** */
%	/* Calculate contribution of only Zonals */
%	/* ********************************************************** */

Anm1(1) = 0;
Anm1(2) = sqrt(3);
Anm2(2) = 0;
Anm2(3) = sqrt(3.75);
as = 0;
at = 0;
au = 0;
ar = 0;
rat1 = 0;
rat2 = 0;
Dnm = 0;
Enm = 0;
Fnm = 0;
Apor = zeros(nmaxp1,1);
Apor(1) = 1;
Apor(2) = RE/rb;

for n = 1:nmax,
    i = n+1;
    an2 = 2*n;
    rat1 = sqrt((an2+3.0)*(((an2+1.0)/n)/(n+2.0))) ;
	rat2 = sqrt((n+1.0)*(((n-1.0)/(an2-1.0))/(an2+1.0))) ;
	Anm1(i+1) = rat1*(u*Anm1(i) - rat2*Anm1(i-1)) ;
	Apor(i) = Apor(i-1)*Apor(2) ;
    if n < mmaxp1
        rat1 = sqrt((an2+5.0)*(((an2+3.0)/n)/(n+4.0))) ;
        rat2 = sqrt((n+3.0)*(((n-1.0)/(an2+1.0))/(an2+3.0))) ;
        Anm2(i+2) = rat1*(u*Anm2(i+1) - rat2*Anm2(i)) ;
    end
    if n < nmaxp1
        rat1 = sqrt(0.5*n*(n+1.0)) ;
        au = au - Apor(i)*rat1*Anm1(i)*(-C(i,1)) ;
        rat2 = sqrt(0.5*((an2+1.0)/(an2+3.0))*(n+1.0)*(n+2.0)) ;
        ar = ar + Apor(i)*rat2*Anm1(i+1)*(-C(i,1)) ;
    end
end

%   /* ********************************************************** */
%	/* Calculate contribution of Tesserals */
%	/* ********************************************************** */

R(1) = 1;
I(1) = 0;
for m = 1:mmax,
    j = m+1;
    am2 = 2*m;
    R(j) = s*R(j-1) - t*I(j-1);
    I(j) = s*I(j-1) + t*R(j-1);
    for l = m:mmax,
        i = l+1;
        Anm(i) = Anm1(i) ;
        Anm1(i) = Anm2(i) ;
    end
    Anm1(mmaxp1) = Anm2(mmaxp1) ;
    for l = m:mmax,
        i = l+1;
        an2 = 2*l;
        if l == m
            Anm2(j+1) = 0.0 ;
            Anm2(j+2) = sqrt((am2+5.0)/(am2+4.0))*Anm1(j+1) ;
        else
            rat1 = sqrt((an2+5.0)*(((an2+3.0)/(l-m))/(l+m+4.0))) ;
            rat2 = sqrt((l+m+3.0)*(((l-m-1.0)/(an2+1.0))/(an2+3.0))) ;
            Anm2(i+2) = rat1*(u*Anm2(i+1) - rat2*Anm2(i)) ;
        end
        Dnm = C(i,j)*R(j) + S(i,j)*I(j) ;
        Enm = C(i,j)*R(j-1) + S(i,j)*I(j-1) ;
        Fnm = S(i,j)*R(j-1) - C(i,j)*I(j-1) ;

        rat1 = sqrt((l+m+1.0)*(l-m)) ;
        rat2 = sqrt(((an2+1.0)/(an2+3.0))*(l+m+1.0)*(l+m+2.0)) ;

        as = as + Apor(i)*m*Anm(i)*Enm ;
        at = at + Apor(i)*m*Anm(i)*Fnm ;
        au = au + Apor(i)*rat1*Anm1(i)*Dnm ;
        ar = ar - Apor(i)*rat2*Anm1(i+1)*Dnm ;
    end
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the spacecraft accelerations in ECEF %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

agx_ECEF = -mur3*x + mur2*(as + ar*s);
agy_ECEF = -mur3*y + mur2*(at + ar*t);
agz_ECEF = -mur3*z + mur2*(au + ar*u);
