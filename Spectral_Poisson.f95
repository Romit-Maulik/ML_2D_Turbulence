!---------------------------------------------------------------------------!
!Spectral accurate Poisson solver
!Periodic, equidistant grid
!Taken from MAE5093 - Github
!Matches with GS
!---------------------------------------------------------------------------!
subroutine solve_poisson(u_org,f_org,nx,ny,dx,dy)
implicit none
integer,intent(in)::nx,ny
double precision,intent(in) ::dx,dy
double precision,intent(in)::f_org(0:nx-1,0:ny-1)
double precision,intent(inout):: u_org(0:nx-1,0:ny-1)
double precision ::pi,Lx,Ly,den
double precision ::kx(0:nx-1),ky(0:ny-1) 
double precision ::data1d(2*nx*ny) 
integer::i,j,k,isign,ndim,nn(2)

!2d data
ndim =2
nn(1)=nx
nn(2)=ny

!1.Find the f coefficient in Fourier space
!assign 1d data array
k=1
do j=0,ny-1  
do i=0,nx-1   
  data1d(k)   =  f_org(i,j)
  data1d(k+1) =  0.0d0    
k = k + 2
end do
end do

!normalize
do k=1,2*nx*ny
data1d(k)=data1d(k)/dfloat(nx*ny)
end do
!inverse fourier transform
isign= -1
call fourn(data1d,nn,ndim,isign)

!2.Solve for u coeeficient in Fourier space
!coefficients
Lx = dfloat(nx)*dx
Ly = dfloat(ny)*dy

!wave numbers (scaled)
pi = 4.0d0*datan(1.0d0)
do i=0,nx/2-1
kx(i)      = (2.0d0*pi/Lx)*dfloat(i)
kx(i+nx/2) = (2.0d0*pi/Lx)*dfloat(i-nx/2)
end do
do j=0,ny/2-1
ky(j)      = (2.0d0*pi/Ly)*dfloat(j)
ky(j+ny/2) = (2.0d0*pi/Ly)*dfloat(j-ny/2)
end do
kx(0) = 1.0d-6 !to eleminate zero division
ky(0) = 1.0d-6 !to eleminate zero division
data1d(1) = 0.0d0
data1d(2) = 0.0d0

!Fourier coefficients for u
k=1
do j=0,ny-1
do i=0,nx-1   
    den = -(kx(i)*kx(i))-(ky(j)*ky(j))
  data1d(k)   =  data1d(k)/den
  data1d(k+1) =  data1d(k+1)/den
k = k + 2
end do
end do

!3. Find u values on physical space
!forward fourier transform
isign= 1
call fourn(data1d,nn,ndim,isign)

!assign 2d array
k=1
do j=0,ny-1
do i=0,nx-1
u_org(i,j)=data1d(k)
k=k+2
end do
end do

return
end


!-----------------------------------------------------------------!
! fft routine from numerical recipes
! ndim: dimension of the transform (i.e.; 2 for 2d problems)
! nn  : number of points in each direction
! data: one-dimensional array including real and imaginary part 
!-----------------------------------------------------------------!
subroutine fourn(data,nn,ndim,isign)
implicit none
integer:: ndim,isign
integer:: nn(ndim)
double precision:: data(*)
double precision:: wr,wi,wpr,wpi,wtemp,theta,tempr,tempi
integer::ntot,n,nrem,nprev,idim,ip1,ip2,ip3,i1,i2,i3
integer::i2rev,i3rev,ibit,ifp1,ifp2,k1,k2

      ntot=1
      do 11 idim=1,ndim
        ntot=ntot*nn(idim)
11    continue
      nprev=1
      do 18 idim=1,ndim
        n=nn(idim)
        nrem=ntot/(n*nprev)
        ip1=2*nprev
        ip2=ip1*n
        ip3=ip2*nrem
        i2rev=1
        do 14 i2=1,ip2,ip1
          if(i2.lt.i2rev)then
            do 13 i1=i2,i2+ip1-2,2
              do 12 i3=i1,ip3,ip2
                i3rev=i2rev+i3-i2
                tempr=data(i3)
                tempi=data(i3+1)
                data(i3)=data(i3rev)
                data(i3+1)=data(i3rev+1)
                data(i3rev)=tempr
                data(i3rev+1)=tempi
12            continue
13          continue
          endif
          ibit=ip2/2
1         if ((ibit.ge.ip1).and.(i2rev.gt.ibit)) then
            i2rev=i2rev-ibit
            ibit=ibit/2
          go to 1
          endif
          i2rev=i2rev+ibit
14      continue
        ifp1=ip1
2       if(ifp1.lt.ip2)then
          ifp2=2*ifp1
          theta=isign*6.28318530717959d0/(ifp2/ip1)
          wpr=-2.d0*dsin(0.5d0*theta)**2
          wpi=dsin(theta)
          wr=1.d0
          wi=0.d0
          do 17 i3=1,ifp1,ip1
            do 16 i1=i3,i3+ip1-2,2
              do 15 i2=i1,ip3,ifp2
                k1=i2
                k2=k1+ifp1
                tempr=sngl(wr)*data(k2)-sngl(wi)*data(k2+1)
                tempi=sngl(wr)*data(k2+1)+sngl(wi)*data(k2)
                data(k2)=data(k1)-tempr
                data(k2+1)=data(k1+1)-tempi
                data(k1)=data(k1)+tempr
                data(k1+1)=data(k1+1)+tempi
15            continue
16          continue
            wtemp=wr
            wr=wr*wpr-wi*wpi+wr
            wi=wi*wpr+wtemp*wpi+wi
17        continue
          ifp1=ifp2
        go to 2
        endif
        nprev=n*nprev
18    continue

return
end
