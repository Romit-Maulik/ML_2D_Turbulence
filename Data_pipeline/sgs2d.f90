program computesgs2d
	implicit none

	integer :: nx, ny, nxc, nyc, nr, i, j, ifl, ns, nf, kfile
	double precision, dimension(:,:), allocatable :: x, y
	double precision, dimension(:,:), allocatable :: w, s, jb
	double precision, dimension(:,:), allocatable :: xc, yc
	double precision, dimension(:,:), allocatable :: wc, sc, jbcc, jbc, lap_c
	double precision, dimension(:,:), allocatable :: s11, s12, s22, u, v, nue, sgs, vel_abs, vel_grad
	double precision, dimension(:,:,:), allocatable :: inv, sgs_models
	double precision :: dx, dy, pi
	character(80):: snapID,corID,filename
	CHARACTER(LEN=80) :: FMT

	!pi
	pi = datan(1.0d0)*4.0d0

	!w,s,j are on fine grid
	!wc, sc, jc are obtained by Fourier cutoff of fine grid quantities
	!jcc is the Jacobian obtained by using wc, sc on coarse grid

	!Number of files
	ns = 0
	nf = 10

	!Resolutions
	nx = 2048
	ny = nx
	!Grid ratio
	nr = 8

	write(corID,'(i5)') nr  !index for coarsening ratio

	!LES resolution
	nxc=nx/nr
	nyc=nxc

	do ifl = ns, nf
		!DNS Grid
		allocate(x(0:nx,0:ny))
		allocate(y(0:nx,0:ny))
		allocate(w(0:nx,0:ny))
		allocate(s(0:nx,0:ny))
		allocate(jb(0:nx,0:ny))


		write(snapID,'(i5)') ifl  !index for time snapshot

		kfile = 500+ifl
		write(filename, "(A5,I0)") "fort.",kfile

		!Reading the DNS data
		!filename = 'Fort.50'// trim(adjustl(snapID)) //'.dat' 

		open(unit=19, file=filename)
		read(19,*)
		read(19,*) 
		do j=0,ny
		do i=0,nx
		read(19,*)x(i,j),y(i,j),w(i,j)
		end do
		end do
		close(19)

		!Calculate Streamfunction/Jacobian on fine grid
		dx = 2.0d0*pi/dfloat(nx)
		dy = dx

		!Calculate streamfunction
		call solve_poisson(s,-w,nx,ny,dx,dy)
		!Calculate Jacobian
		call jacobian_calc(w,s,nx,ny,jb,dx,dy)

				!LES Grid
		allocate(xc(0:nxc,0:nyc))
		allocate(yc(0:nxc,0:nyc))

		!direct coarsening:old way
		do j=0,nyc
		do i=0,nxc
		xc(i,j)=x(nr*i,nr*j)
		yc(i,j)=y(nr*i,nr*j)
		end do
		end do

		deallocate(x,y)

		allocate(wc(0:nxc,0:nyc))
		allocate(sc(0:nxc,0:nyc))
		allocate(jbc(0:nxc,0:nyc))
		
		!fft coarsening:correct way
		call coarsen2d(nx,ny,nxc,nyc,w,wc)
		call coarsen2d(nx,ny,nxc,nyc,s,sc)
		call coarsen2d(nx,ny,nxc,nyc,jb,jbc)

		allocate(lap_c(0:nxc,0:nyc))
		call laplacian_calc(nxc,nyc,dx,dy,wc,lap_c)

		deallocate(w,s,jb)

		allocate(jbcc(0:nxc,0:nyc))
		
		!Compute Jacobian on coarse grid
		dx = 2.0d0*pi/dfloat(nxc)
		dy = dx
		call jacobian_calc(wc,sc,nxc,nyc,jbcc,dx,dy)

		allocate(inv(0:nxc,0:nyc,1:2))
		allocate(u(0:nxc,0:nyc))
		allocate(v(0:nxc,0:nyc))
		
		!Calculate kernel calculations
		call kernel_calculation(nxc,nyc,wc,sc,dx,dy,inv,u,v)

		allocate(s11(0:nxc,0:nyc))
		allocate(s12(0:nxc,0:nyc))
		allocate(s22(0:nxc,0:nyc))
		allocate(vel_abs(0:nxc,0:nyc))
		allocate(vel_grad(0:nxc,0:nyc))
		
		!Calculate strain related stuff
		call strain_calc(nxc,nyc,dx,dy,u,v,s11,s12,s22,vel_abs,vel_grad)

		allocate(nue(0:nxc,0:nyc))
		allocate(sgs(0:nxc,0:nyc))
		!Calculate eddy viscosities
		do j = 0,nyc
			do i = 0,nxc
				nue(i,j) = (jbcc(i,j)-jbc(i,j))/lap_c(i,j)
				sgs(i,j) = (jbcc(i,j)-jbc(i,j))
			end do
		end do

		! Calculate Smag, Leith, AD estimates
		allocate(sgs_models(0:nxc,0:nyc,1:3))
		call model_estimate_calc(nxc,nyc,wc,sc,dx,dy,sgs_models)

		!Output coarse quantities
		write(filename, "(A7,I0)") "Source.",kfile
		open(unit=19, file=filename)!Writing source as well as subgrid estimated Re loss
		do j=0,nyc
		do i=0,nxc
		write(19,*)wc(i,j),sc(i,j),inv(i,j,1),inv(i,j,2),lap_c(i,j),vel_abs(i,j),sgs_models(i,j,1),sgs_models(i,j,2),&
		sgs_models(i,j,3),sgs(i,j)
		end do
		end do
		close(19)

		deallocate(jbc,jbcc)
		deallocate(xc,yc,wc,sc,inv,lap_c,u,v,nue,sgs,s11,s12,s22,vel_abs,vel_grad)
		deallocate(sgs_models)

	end do


end program


!------------------------------------------------------------------!
!------------------------------------------------------------------!
!Strain rate calculations
!------------------------------------------------------------------!
!------------------------------------------------------------------!
subroutine strain_calc(nx,ny,dx,dy,u,v,s11,s12,s22,vel_abs,vel_grad)
	implicit none

	integer :: nx, ny, i, j
	real*8 :: dx, dy
	real*8, dimension(0:nx,0:ny) :: u,v,s11, s12, s22, vel_grad, vel_abs
	real*8, dimension(:,:),allocatable :: utemp,vtemp
	real*8 :: dudx, dudy, dvdx, dvdy

	allocate(utemp(-1:nx,-1:ny))
	allocate(vtemp(-1:nx,-1:ny))


	do j = 0,ny-1
	do i = 0,nx-1
	utemp(i,j) = u(i,j)
	vtemp(i,j) = v(i,j)
	end do
	end do

	call periodic_bc_update_for_jacobian(nx,ny,utemp)
	call periodic_bc_update_for_jacobian(nx,ny,vtemp)

	do j = 0,ny-1
	do i = 0,nx-1

	dudx = (utemp(i+1,j)-utemp(i-1,j))/(2.0d0*dx)
	dudy = (utemp(i,j+1)-utemp(i,j-1))/(2.0d0*dy)
	dvdx = (vtemp(i+1,j)-vtemp(i-1,j))/(2.0d0*dx)
	dvdy = (vtemp(i,j+1)-vtemp(i,j-1))/(2.0d0*dy)

	s11(i,j) = dudx
	s22(i,j) = dvdy
	s12(i,j) = 0.5d0*(dudy + dvdx)

	vel_grad(i,j) = dsqrt(dudx**2 + dvdy**2)
	vel_abs(i,j) = dsqrt(utemp(i,j)**2 + vtemp(i,j)**2)
	end do
	end do

	deallocate(utemp,vtemp)




	return
end


!-----------------------------------------------------------------!
!Compute coarsening using FFT
!-----------------------------------------------------------------!
subroutine coarsen2d(nx,ny,nxc,nyc,u,uc)
implicit none
integer::nx,ny
integer::nxc,nyc
double precision ::u(0:nx,0:ny)
double precision ::uc(0:nxc,0:nyc)
integer::i,j,p
double precision,dimension(:),allocatable:: data1d
double precision,dimension(:),allocatable:: data1c
complex*16, dimension(:,:), allocatable::gm, gc
integer,parameter::ndim=2
integer::nn(ndim),isign
double precision ::temp

allocate(data1d(2*nx*ny))
allocate(data1c(2*nxc*nyc))

!finding fourier coefficients
p=1
do j=0,ny-1
do i=0,nx-1
data1d(p)  =u(i,j)
data1d(p+1)=0.0d0
p=p+2
end do
end do

!normalize
temp=1.0d0/dfloat(nx*ny)
do p=1,2*nx*ny
data1d(p)=data1d(p)*temp
end do

!invese fourier transform
nn(1)= nx
nn(2)= ny
isign=-1
call fourn(data1d,nn,ndim,isign)

!in the following: gm=fine data & gc=coarsen data

!finding fourier coefficients: real and imaginary
allocate(gm(0:nx-1,0:ny-1))
p=1
do j=0,ny-1
do i=0,nx-1
gm(i,j)= dcmplx(data1d(p),data1d(p+1))  
p=p+2
end do
end do

!coarsen:
allocate(gc(0:nxc-1,0:nyc-1))

do j=0,nyc/2
	do i=0,nxc/2
	gc(i,j) = gm(i,j)
	end do
	do i=nxc/2+1,nxc-1
	gc(i,j) = gm(i+(nx-nxc),j)
	end do
end do

do j=nyc/2+1,nyc-1
	do i=0,nxc/2
	gc(i,j) = gm(i,j+(ny-nyc))
	end do
	do i=nxc/2+1,nxc-1
	gc(i,j) = gm(i+(nx-nxc),j+(ny-nyc))
	end do
end do


!coarsening
p=1
do j=0,nyc-1
do i=0,nxc-1
data1c(p)=dreal(gc(i,j))
data1c(p+1)=dimag(gc(i,j))
p=p+2
end do
end do

!forward fourier transform
nn(1)= nxc
nn(2)= nyc
isign= 1
call fourn(data1c,nn,ndim,isign)

p=1
do j=0,nyc-1
do i=0,nxc-1
uc(i,j) = data1c(p)
p=p+2
end do
end do

!Periodic BCs for uc
do j = 0,nyc-1
	uc(nxc,j) = uc(0,j)
end do

do i = 0,nxc
	uc(i,nyc) = uc(i,0)
end do



deallocate(data1d,data1c,gc,gm)

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
real*8:: data(*)
real*8:: wr,wi,wpr,wpi,wtemp,theta,tempr,tempi
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


!-------------------------------------------------------------------------!
!-------------------------------------------------------------------------!
!Subroutine for calculation of gradients
!-------------------------------------------------------------------------!
!-------------------------------------------------------------------------!
subroutine gradient_calc(nx,ny,omega,psi,dx,dy,inv)
!$ use omp_lib
implicit none

integer :: nx,ny,i,j,k
double precision, dimension(0:nx,0:ny) :: omega, psi
double precision, dimension(0:nx,0:ny,1:10) :: inv
double precision :: dx,dy,dwdx,dwdy,dsdx,dsdy
double precision, dimension(:,:),allocatable :: omega_new,psi_new

allocate(psi_new(-1:nx,-1:ny))
allocate(omega_new(-1:nx,-1:ny))


do j = 0,ny-1
do i = 0,nx-1
psi_new(i,j) = psi(i,j)
omega_new(i,j) = omega(i,j)
end do
end do

call periodic_bc_update_for_jacobian(nx,ny,psi_new)
call periodic_bc_update_for_jacobian(nx,ny,omega_new)
!Use CS 2nd order

do j = 0,ny-1
do i = 0,nx-1

dwdx = (omega_new(i+1,j)-omega_new(i-1,j))/(2.0d0*dx)
dwdy = (omega_new(i,j+1)-omega_new(i,j-1))/(2.0d0*dy)
dsdx = (psi_new(i+1,j)-psi_new(i-1,j))/(2.0d0*dx)
dsdy = (psi_new(i,j+1)-psi_new(i,j-1))/(2.0d0*dy)


inv(i,j,1) = dwdx*dwdx
inv(i,j,2) = dwdx*dwdy
inv(i,j,3) = dwdx*dsdx
inv(i,j,4) = dwdx*dsdy
inv(i,j,5) = dwdy*dwdy
inv(i,j,6) = dwdy*dsdx
inv(i,j,7) = dwdy*dsdy
inv(i,j,8) = dsdx*dsdx
inv(i,j,9) = dsdx*dsdy
inv(i,j,10) = dsdy*dsdy

end do
end do

!Periodicity
do k = 1,10

do j = 0,ny-1
inv(nx,j,k) = inv(0,j,k)
end do

do i = 0,nx
inv(i,ny,k) = inv(i,0,k)
end do

end do

deallocate(omega_new,psi_new)

return
end



!-------------------------------------------------------------------------!
!-------------------------------------------------------------------------!
!Subroutine for calculation of turbulence model estimates
!call model_estimate_calc(nx,ny,omega,psi,dx,dy,sgs_models)
!-------------------------------------------------------------------------!
!-------------------------------------------------------------------------!
subroutine model_estimate_calc(nx,ny,omega,psi,dx,dy,inv)
implicit none

integer :: nx,ny,i,j,k
double precision, dimension(0:nx,0:ny) :: omega, psi
double precision, dimension(0:nx,0:ny,1:3) :: inv
double precision :: dx,dy, c_turb, lap_val, sigma
double precision, dimension(:,:),allocatable :: omega_new,psi_new, source
double precision, dimension(:,:),allocatable :: d2sdxdy, d2sdx, d2sdy, dsdy
double precision, dimension(:,:),allocatable :: dwdx, dwdy

allocate(psi_new(-1:nx,-1:ny))
allocate(omega_new(-1:nx,-1:ny))

do j = 0,ny-1
do i = 0,nx-1
psi_new(i,j) = psi(i,j)
omega_new(i,j) = omega(i,j)
end do
end do

call periodic_bc_update_for_jacobian(nx,ny,psi_new)
call periodic_bc_update_for_jacobian(nx,ny,omega_new)

!Calculating Smag Turbulence model invariants
allocate(d2sdxdy(0:nx-1,0:ny-1))
allocate(d2sdx(0:nx-1,0:ny-1))
allocate(d2sdy(0:nx-1,0:ny-1))
allocate(dsdy(-1:nx,-1:ny))

do j = 0,ny-1
	do i = 0,nx-1
		dsdy(i,j) = (psi_new(i,j+1)-psi_new(i,j-1))/(2.0d0*dy)
		d2sdx(i,j) = (psi_new(i+1,j)+psi_new(i-1,j)-2.0d0*psi_new(i,j))/(dx*dx)
		d2sdy(i,j) = (psi_new(i,j+1)+psi_new(i,j-1)-2.0d0*psi_new(i,j))/(dy*dy)
	end do
end do

call periodic_bc_update_for_jacobian(nx,ny,dsdy)

do j = 0,ny-1
	do i = 0,nx-1
		d2sdxdy(i,j) = (dsdy(i+1,j)-dsdy(i-1,j))/(2.0d0*dx)
	end do
end do

!Smag invariant
do j = 0,ny-1
	do i = 0,nx-1
		inv(i,j,1) = dsqrt(4.0d0*d2sdxdy(i,j)**2 + (d2sdx(i,j)-d2sdy(i,j))**2)
	end do
end do

deallocate(d2sdxdy,dsdy,d2sdx,d2sdy)

!Calculating Leith turbulence model invariants
allocate(dwdy(0:nx-1,0:ny-1))
allocate(dwdx(0:nx-1,0:ny-1))
do j = 0,ny-1
do i = 0,nx-1
dwdy(i,j) = (omega_new(i,j+1)-omega_new(i,j-1))/(2.0d0*dy)
dwdx(i,j) = (omega_new(i+1,j)-omega_new(i-1,j))/(2.0d0*dx)
end do
end do

!Leith invariant
do j = 0,ny-1
	do i = 0,nx-1
		inv(i,j,2) = dsqrt(dwdx(i,j)**2 + dwdy(i,j)**2)
	end do
end do

deallocate(dwdx,dwdy)

!Combining invariants with turbulence coefficients and laplacian
c_turb = 1.0d0*dx
	do j = 0,ny-1
		do i = 0,nx-1
			lap_val = (omega_new(i+1,j)+omega_new(i-1,j)-2.0d0*omega_new(i,j))/(dx*dx)
			lap_val = lap_val + (omega_new(i,j+1)+omega_new(i,j-1)-2.0d0*omega_new(i,j))/(dy*dy)
			inv(i,j,1) = c_turb*c_turb*inv(i,j,1)*lap_val
			inv(i,j,2) = c_turb*c_turb*c_turb*inv(i,j,2)*lap_val
		end do
	end do

! Call AD to get sgs_ad
sigma = 0.0d0 ! Dummy placeholder
allocate(source(0:nx,0:ny))
call approximate_deconvolution(omega,psi,source,nx,ny,dx,dy,sigma)

do j = 0,ny-1
	do i = 0,nx-1
		inv(i,j,3) = source(i,j)
	end do
end do

deallocate(source)

!Periodicity
do k = 1,3
	do j = 0,ny-1
	inv(nx,j,k) = inv(0,j,k)
	end do

	do i = 0,nx
	inv(i,ny,k) = inv(i,0,k)
	end do
end do

deallocate(omega_new,psi_new)

return
end


!-------------------------------------------------------------------------!
!-------------------------------------------------------------------------!
!Subroutine for calculation of grid resolved kernels
!-------------------------------------------------------------------------!
!-------------------------------------------------------------------------!
subroutine kernel_calculation(nx,ny,omega,psi,dx,dy,inv,u,v)
!$ use omp_lib
implicit none

integer :: nx,ny,i,j,k
double precision, dimension(0:nx,0:ny) :: omega, psi, u, v
double precision, dimension(0:nx,0:ny,1:2) :: inv
double precision :: dx,dy, c_turb, lap_val
double precision, dimension(:,:),allocatable :: omega_new,psi_new
double precision, dimension(:,:),allocatable :: d2sdxdy, d2sdx, d2sdy, dsdy, dsdx
double precision, dimension(:,:),allocatable :: dwdx, dwdy

allocate(psi_new(-1:nx,-1:ny))
allocate(omega_new(-1:nx,-1:ny))


do j = 0,ny-1
do i = 0,nx-1
psi_new(i,j) = psi(i,j)
omega_new(i,j) = omega(i,j)
end do
end do


call periodic_bc_update_for_jacobian(nx,ny,psi_new)
call periodic_bc_update_for_jacobian(nx,ny,omega_new)

!Calculating Smag Turbulence model invariants
allocate(d2sdxdy(0:nx-1,0:ny-1))
allocate(d2sdx(0:nx-1,0:ny-1))
allocate(d2sdy(0:nx-1,0:ny-1))
allocate(dsdy(-1:nx,-1:ny))
allocate(dsdx(0:nx-1,0:ny-1))

do j = 0,ny-1
	do i = 0,nx-1
		dsdy(i,j) = (psi_new(i,j+1)-psi_new(i,j-1))/(2.0d0*dy)
		dsdx(i,j) = (psi_new(i+1,j)-psi_new(i-1,j))/(2.0d0*dx)
		d2sdx(i,j) = (psi_new(i+1,j)+psi_new(i-1,j)-2.0d0*psi_new(i,j))/(dx*dx)
		d2sdy(i,j) = (psi_new(i,j+1)+psi_new(i,j-1)-2.0d0*psi_new(i,j))/(dy*dy)
	end do
end do

do j = 0,ny-1
	do i = 0, nx-1
		u(i,j) = dsdy(i,j)
		v(i,j) = -dsdx(i,j)
	end do
end do

call periodic_bc_update_for_jacobian(nx,ny,dsdy)

do j = 0,ny-1
	do i = 0,nx-1
		d2sdxdy(i,j) = (dsdy(i+1,j)-dsdy(i-1,j))/(2.0d0*dx)
	end do
end do

!Smag invariant
do j = 0,ny-1
	do i = 0,nx-1
		inv(i,j,1) = dsqrt(4.0d0*d2sdxdy(i,j)**2 + (d2sdx(i,j)-d2sdy(i,j))**2)
	end do
end do

deallocate(d2sdxdy,dsdy,d2sdx,d2sdy,dsdx)

!Calculating Leith turbulence model invariants
allocate(dwdy(0:nx-1,0:ny-1))
allocate(dwdx(0:nx-1,0:ny-1))
do j = 0,ny-1
do i = 0,nx-1
dwdy(i,j) = (omega_new(i,j+1)-omega_new(i,j-1))/(2.0d0*dy)
dwdx(i,j) = (omega_new(i+1,j)-omega_new(i-1,j))/(2.0d0*dx)
end do
end do

!Leith invariant
do j = 0,ny-1
	do i = 0,nx-1
		inv(i,j,2) = dsqrt(dwdx(i,j)**2 + dwdy(i,j)**2)
	end do
end do

deallocate(dwdx,dwdy)

!Periodicity
do k = 1,2
	do j = 0,ny-1
	inv(nx,j,k) = inv(0,j,k)
	end do

	do i = 0,nx
	inv(i,ny,k) = inv(i,0,k)
	end do
end do

deallocate(omega_new,psi_new)

return
end

!-------------------------------------------------------------------------!
!-------------------------------------------------------------------------!
!Subroutine for calculation of Jacobian
!-------------------------------------------------------------------------!
!-------------------------------------------------------------------------!
subroutine jacobian_calc(omega,psi,nx,ny,jc,dx,dy)
!$ use omp_lib
implicit none

integer :: nx, ny, i, j
double precision, dimension(0:nx,0:ny) :: omega, psi, jc
double precision, dimension(:,:), allocatable :: psi_new, omega_new
double precision :: jj1, jj2, jj3, dx, dy

allocate(psi_new(-1:nx,-1:ny))
allocate(omega_new(-1:nx,-1:ny))

do j = 0,ny-1
do i = 0,nx-1
psi_new(i,j) = psi(i,j)
omega_new(i,j) = omega(i,j)
end do
end do

call periodic_bc_update_for_jacobian(nx,ny,psi_new)
call periodic_bc_update_for_jacobian(nx,ny,omega_new)

!$OMP PARALLEL DO
do j = 0,ny-1
do i = 0,nx-1


jj1 = 1.0/(4.0*dx*dy) * ((omega_new(i+1,j)-omega_new(i-1,j)) * (psi_new(i,j+1) - psi_new(i,j-1)) &
			- (omega_new(i,j+1)-omega_new(i,j-1)) * (psi_new(i+1,j) - psi_new(i-1,j)))

jj2 = 1.0 / (4.0 * dx * dy) * (omega_new(i+1, j) * (psi_new(i+1, j+1) - psi_new(i+1, j-1)) &
                                         - omega_new(i-1, j) * (psi_new(i-1, j+1) - psi_new(i-1, j-1)) &
                                         - omega_new(i, j+1) * (psi_new(i+1, j+1) - psi_new(i-1, j+1)) &
                                         + omega_new(i, j-1) * (psi_new(i+1, j-1) - psi_new(i-1, j-1)) &
                                          )

jj3 = 1.0 / (4.0 * dx * dy) * (omega_new(i+1, j+1) * (psi_new(i, j+1) - psi_new(i+1, j)) &
                                        -  omega_new(i-1, j-1) * (psi_new(i-1, j) - psi_new(i, j-1)) &
                                        -  omega_new(i-1, j+1) * (psi_new(i, j+1) - psi_new(i-1, j)) &
                                        +  omega_new(i+1, j-1) * (psi_new(i+1, j) - psi_new(i, j-1)) &
                                          )

jc(i, j) = (jj1 + jj2 + jj3)/3.0

end do
end do

!$OMP END PARALLEL DO

do j=0,ny-1
	jc(nx,j) = jc(0,j)
end do

do i = 0,nx
	jc(i,ny) = jc(i,0)
end do

deallocate(psi_new,omega_new)

return
end

!-------------------------------------------------------------------------!
!-------------------------------------------------------------------------!
!Subroutine for calculation of pointwise laplacian
!call laplacian_calc(nxc,nyc,dx,dy,wc,lap_c)
!-------------------------------------------------------------------------!
!-------------------------------------------------------------------------!
subroutine laplacian_calc(nx,ny,dx,dy,omega,laplacian)
!$ use omp_lib
implicit none

integer :: nx, ny, i, j
double precision, dimension(0:nx,0:ny) :: omega, laplacian
double precision, dimension(:,:), allocatable :: omega_new
double precision :: dx, dy

allocate(omega_new(-1:nx,-1:ny))

do j = 0,ny-1
do i = 0,nx-1
omega_new(i,j) = omega(i,j)
end do
end do

!Same as Jacobian update
call periodic_bc_update_for_jacobian(nx,ny,omega_new)

!$OMP PARALLEL DO
do j = 0,ny-1
do i = 0,nx-1


laplacian(i,j) = (omega_new(i+1,j)+omega_new(i-1,j)-2.0d0*omega_new(i,j))/(dx*dx)
laplacian(i,j) = laplacian(i,j) + (omega_new(i,j+1)+omega_new(i,j-1)-2.0d0*omega_new(i,j))/(dy*dy)

end do
end do

!$OMP END PARALLEL DO

do j=0,ny-1
	laplacian(nx,j) = laplacian(0,j)
end do

do i = 0,nx
	laplacian(i,ny) = laplacian(i,0)
end do

deallocate(omega_new)

return
end


!---------------------------------------------------------------------------!
!subroutine - boundary condition update
!---------------------------------------------------------------------------!
subroutine periodic_bc_update_for_jacobian(nx,ny,u)
implicit none

integer :: nx, ny, i, j
double precision, dimension(-1:nx,-1:ny) :: u

do i = 0,nx-1
u(i,-1) = u(i,ny-1)
u(i,ny) = u(i,0)
end do

do j = -1,ny
u(-1,j) = u(nx-1,j)
u(nx,j) = u(0,j)
end do

end subroutine

!---------------------------------------------------------------------------!
!Spectral accurate Poisson solver
!Periodic, equidistant grid
!---------------------------------------------------------------------------!
subroutine solve_poisson(u_org,f_org,nx,ny,dx,dy)
implicit none
integer,intent(in)::nx,ny
double precision,intent(in) ::dx,dy
double precision,intent(in)::f_org(0:nx,0:ny)
double precision,intent(inout):: u_org(0:nx,0:ny)
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

!Periodic BCs for u_org
do j = 0,ny-1
	u_org(nx,j) = u_org(0,j)
end do

do i = 0,nx
	u_org(i,ny) = u_org(i,0)
end do

return
end

!----------------------------------------------------------------------------!
!Subroutine for source term calculation using AD
!Validated - used for ML-AD-SFS predictions
!----------------------------------------------------------------------------!
subroutine approximate_deconvolution(omega,psi,source,nx,ny,dx,dy,sigma)

implicit none

integer, intent(in) :: nx, ny
double precision, intent(in) :: dx, dy, sigma
double precision, dimension(0:nx,0:ny), intent(in) :: omega, psi
double precision, dimension(0:nx,0:ny), intent(out) :: source
double precision, dimension(:,:),allocatable :: jcf,jcad,psi_ad,omega_ad
integer :: i,j

allocate(jcf(0:nx,0:ny))
allocate(jcad(0:nx,0:ny))

!Compute Jacobian of filtered variables
call jacobian_calc(omega,psi,nx,ny,jcf,dx,dy)

!AD process
allocate(psi_ad(0:nx,0:ny))
allocate(omega_ad(0:nx,0:ny))

call adm(nx,ny,psi,psi_ad, sigma, 3)
call adm(nx,ny,omega,omega_ad, sigma, 3)

!Compute Jacobian of deconvolved variables
call jacobian_calc(omega_ad,psi_ad,nx,ny,jcad,dx,dy)

!Compute filtered AD jacobian
call filter_trapezoidal(jcad,nx,ny)

do j = 0,ny
	do i = 0,nx
		source(i,j) = jcf(i,j)-jcad(i,j)
	end do
end do

deallocate(jcf,jcad,omega_ad,psi_ad)

return
end

!-----------------------------------------------------------------!
!Trapezoidal filter
!-----------------------------------------------------------------!
subroutine filter_trapezoidal(q_org,nx,ny)
implicit none
integer,intent(in)::nx,ny
double precision, intent(inout) :: q_org(0:nx,0:ny)
integer :: i,j
double precision, dimension(:,:), allocatable :: wt
double precision::dd

allocate(wt(-1:nx,-1:ny))

do j = 0,ny-1
do i = 0,nx-1
    wt(i,j) = q_org(i,j)
end do
end do

call periodic_bc_update_for_jacobian(nx,ny,wt)

dd=1.0d0/16.0d0

do j=0,ny-1
do i=0,nx-1
q_org(i,j) = dd*(4.0d0*wt(i,j) &
       + 2.0d0*(wt(i+1,j) + wt(i-1,j) + wt(i,j+1) + wt(i,j-1)) &
	   + wt(i+1,j+1) + wt(i-1,j-1) + wt(i+1,j-1) + wt(i-1,j+1))
end do
end do

deallocate(wt)

return
end 


!-------------------------------------------------------------------------!
!Subroutine for approximate deconvolution of omega and psi variables
!Utilizes 3 iterative resubstitutions
!Validated
!-------------------------------------------------------------------------!
subroutine adm(nx,ny,uf,u_ad,sigma,ad_iter)
implicit none

integer :: nx, ny, k, ad_iter
double precision :: sigma
double precision,dimension(0:nx,0:ny) :: uf, u_ad
double precision, dimension(:,:),allocatable :: utemp

allocate(utemp(0:nx,0:ny))

!Initialize as filtered variable
u_ad = uf

do k = 1,ad_iter

utemp = u_ad
! call filter_gaussian(utemp,nx,ny,sigma)
call filter_trapezoidal(utemp,nx,ny)
u_ad = u_ad + (uf - utemp)

end do

deallocate(utemp)

return
end