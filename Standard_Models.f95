!-----------------------------------------------------------------------------------!
!Implemented natively - Romit Maulik
!Filtering (Gaussian) - using definition in classical image processing literature
!Refer Wolfram Mathematica notebook in Kappa_SFS_Predictions for kernel development
!Validated
!-----------------------------------------------------------------------------------!
subroutine filter_gaussian(q_org,nx,ny,sigma)
!$ use omp_lib

implicit none
integer,intent(in)::nx,ny
double precision, intent(inout) :: q_org(0:nx-1,0:ny-1)
double precision, intent(in) :: sigma

integer ::i,j,k
double precision,allocatable:: q(:,:),u(:,:),v(:,:)
double precision :: sumval,pi
double precision,dimension(-3:3) :: kernel

pi = datan(1.0d0)*4.0d0

!Obtain kernel for convolution from classical Gaussian used in image processing literature
sumval = 0.0d0
do i = -3,3
kernel(i) = dexp(-(dfloat(i*i)/(2.0d0*sigma**2)))/dsqrt(2.0d0*(sigma**2)*pi)
sumval = sumval + kernel(i)
end do

!Normalize the kernel to unity
do i =-3,3
kernel(i) = kernel(i)/sumval
end do

allocate(q(-3:nx+2,-3:ny+2))
allocate(u(-3:nx+2,-3:ny+2))
allocate(v(-3:nx+2,-3:ny+2))

do j=0,ny-1
do i=0,nx-1
u(i,j) = q_org(i,j)
end do
end do

call periodic_bc_update_for_filter(nx,ny,u)

!$OMP PARALLEL DO
!filter in y
do j=0,ny-1
do i=0,nx-1

sumval = 0.0d0
do k = -3,3
sumval = sumval + kernel(k)*u(i,j+k)
end do
v(i,j) = sumval

end do
end do
!$OMP END PARALLEL DO

call periodic_bc_update_for_filter(nx,ny,v)

!filter in x
!$OMP PARALLEL DO
do j=0,ny-1
do i=0,nx-1


sumval = 0.0d0
do k = -3,3
sumval = sumval + kernel(k)*v(i,j+k)
end do
q(i,j) = sumval
              
end do
end do
!$OMP END PARALLEL DO


call periodic_bc_update_for_filter(nx,ny,q)

do j = 0,ny-1
     do i = 0,nx-1
          q_org(i,j) = q(i,j)
     end do
end do

deallocate(u,v,q)

return
end


!---------------------------------------------------------------------------!
!subroutine - boundary condition update for +-3
!Validated
!---------------------------------------------------------------------------!
subroutine periodic_bc_update_for_filter(nx,ny,u)
implicit none

integer :: nx, ny, i, j
double precision, dimension(-3:nx+2,-3:ny+2) :: u

do i = 0,nx-1
u(i,-1) = u(i,ny-1)
u(i,-2) = u(i,ny-2)
u(i,-3) = u(i,ny-3)


u(i,ny) = u(i,0)
u(i,ny+1) = u(i,1)
u(i,ny+2) = u(i,2)
end do

do j = -3,ny+2
u(-1,j) = u(nx-1,j)
u(-2,j) = u(nx-2,j)
u(-3,j) = u(nx-3,j)


u(nx,j) = u(0,j)
u(nx+1,j) = u(1,j)
u(nx+2,j) = u(2,j)

end do

end subroutine


!----------------------------------------------------------------------------!
!Subroutine for source term calculation using AD
!Validated - used for ML-AD-SFS predictions
!----------------------------------------------------------------------------!
subroutine approximate_deconvolution(omega,psi,source,nx,ny,dx,dy,sigma)
!$ use omp_lib

implicit none

integer, intent(in) :: nx, ny
double precision, intent(in) :: dx, dy, sigma
double precision, dimension(0:nx-1,0:ny-1), intent(in) :: omega, psi
double precision, dimension(0:nx-1,0:ny-1), intent(out) :: source
double precision, dimension(:,:),allocatable :: jcf,jcad,psi_ad,omega_ad
integer :: i,j

allocate(jcf(0:nx-1,0:ny-1))
allocate(jcad(0:nx-1,0:ny-1))

!Compute Jacobian of filtered variables
call jacobian_calc(omega,psi,nx,ny,jcf,dx,dy)

!AD process
allocate(psi_ad(0:nx-1,0:ny-1))
allocate(omega_ad(0:nx-1,0:ny-1))

call adm(nx,ny,psi,psi_ad, sigma, 3)
call adm(nx,ny,omega,omega_ad, sigma, 3)

!Compute Jacobian of deconvolved variables
call jacobian_calc(omega_ad,psi_ad,nx,ny,jcad,dx,dy)

!Compute filtered AD jacobian
call filter_trapezoidal(jcad,nx,ny)

!$OMP PARALLEL DO
do j = 0,ny-1
	do i = 0,nx-1
		source(i,j) = jcf(i,j)-jcad(i,j)
	end do
end do
!$OMP END PARALLEL DO

deallocate(jcf,jcad,omega_ad,psi_ad)

return
end


!----------------------------------------------------------------------------!
!Subroutine for source term calculation using Layton method (N AD = 1)
!----------------------------------------------------------------------------!
subroutine layton(omega,psi,source,nx,ny,dx,dy,sigma)
!$ use omp_lib

implicit none

integer, intent(in) :: nx, ny
double precision, intent(in) :: dx, dy, sigma
double precision, dimension(0:nx-1,0:ny-1), intent(in) :: omega, psi
double precision, dimension(0:nx-1,0:ny-1), intent(out) :: source
double precision, dimension(:,:),allocatable :: jcf,jcad,psi_ad,omega_ad
integer :: i,j

allocate(jcf(0:nx-1,0:ny-1))
allocate(jcad(0:nx-1,0:ny-1))

!Compute Jacobian of filtered variables
call jacobian_calc(omega,psi,nx,ny,jcf,dx,dy)

!AD process
allocate(psi_ad(0:nx-1,0:ny-1))
allocate(omega_ad(0:nx-1,0:ny-1))

call adm(nx,ny,psi,psi_ad, sigma, 1)
call adm(nx,ny,omega,omega_ad, sigma, 1)

!Compute Jacobian of deconvolved variables
call jacobian_calc(omega_ad,psi_ad,nx,ny,jcad,dx,dy)

!Compute filtered AD jacobian
call filter_trapezoidal(jcad,nx,ny)

!$OMP PARALLEL DO
do j = 0,ny-1
	do i = 0,nx-1
		source(i,j) = jcf(i,j)-jcad(i,j)
	end do
end do
!$OMP END PARALLEL DO

deallocate(jcf,jcad,omega_ad,psi_ad)

return
end


!----------------------------------------------------------------------------!
!Subroutine for source term calculation using Layton method (N AD = 1)
!----------------------------------------------------------------------------!
subroutine bardina(omega,psi,source,nx,ny,dx,dy,sigma)
!$ use omp_lib

implicit none

integer, intent(in) :: nx, ny
double precision, intent(in) :: dx, dy, sigma
double precision, dimension(0:nx-1,0:ny-1), intent(in) :: omega, psi
double precision, dimension(0:nx-1,0:ny-1), intent(out) :: source
double precision, dimension(:,:),allocatable :: jcf,jcff,psi_f,omega_f
integer :: i,j

allocate(jcf(0:nx-1,0:ny-1))
allocate(jcff(0:nx-1,0:ny-1))
allocate(psi_f(0:nx-1,0:ny-1))
allocate(omega_f(0:nx-1,0:ny-1))


!Compute Jacobian of grid-filtered variables
call jacobian_calc(omega,psi,nx,ny,jcf,dx,dy)
call filter_trapezoidal(jcff,nx,ny)

!Compute Jacobian of explicit-filtered variables
psi_f = psi
omega_f = omega

call filter_trapezoidal(psi_f,nx,ny)
call filter_trapezoidal(omega_f,nx,ny)

call jacobian_calc(omega_f,psi_f,nx,ny,jcf,dx,dy)


!$OMP PARALLEL DO
do j = 0,ny-1
	do i = 0,nx-1
		source(i,j) = jcf(i,j)-jcff(i,j)
	end do
end do
!$OMP END PARALLEL DO

deallocate(jcf,jcff,omega_f,psi_f)

return
end



!-------------------------------------------------------------------------!
!Subroutine for calculation of Jacobian
!Validated
!-------------------------------------------------------------------------!
subroutine jacobian_calc(omega,psi,nx,ny,jc,dx,dy)
!$ use omp_lib
implicit none

integer :: nx, ny, i, j
double precision, dimension(0:nx-1,0:ny-1) :: omega, psi, jc
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

deallocate(psi_new,omega_new)

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
double precision,dimension(0:nx-1,0:ny-1) :: uf, u_ad
double precision, dimension(:,:),allocatable :: utemp

allocate(utemp(0:nx-1,0:ny-1))

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



!---------------------------------------------------------------------------!
!subroutine - boundary condition update
!Validated
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


!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
!Subroutine for smagorinsky based SGS prediction
!Validated
!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
subroutine smag_source_term(omega,psi,sgs,nx,ny,dx,dy)

!$ use omp_lib
implicit none
integer, intent(in) :: nx,ny
integer :: i,j
double precision,dimension(0:nx-1,0:ny-1),intent(in) :: omega, psi
double precision, intent(in) :: dx, dy
double precision,dimension(0:nx-1,0:ny-1),intent(out) :: sgs
double precision, dimension(:,:), allocatable :: omega_new, psi_new
double precision, dimension(:,:), allocatable :: dsdy, d2sdx, d2sdy, d2sdxdy
double precision :: c_turb, lap_val

allocate(omega_new(-1:nx,-1:ny))
allocate(psi_new(-1:nx,-1:ny))

do j = 0,ny-1
  do i = 0,nx-1
    omega_new(i,j) = (omega(i,j))
    psi_new(i,j) = psi(i,j)
  end do
end do

!BC Update
call periodic_bc_update(nx,ny,omega_new)
call periodic_bc_update(nx,ny,psi_new)

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

call periodic_bc_update(nx,ny,dsdy)!Need for a second derivative for this quantity

do j = 0,ny-1
  do i = 0,nx-1
    d2sdxdy(i,j) = (dsdy(i+1,j)-dsdy(i-1,j))/(2.0d0*dx)
  end do
end do

!Smag invariant
do j = 0,ny-1
  do i = 0,nx-1
    sgs(i,j) = dsqrt(4.0d0*d2sdxdy(i,j)**2 + (d2sdx(i,j)-d2sdy(i,j))**2)
  end do
end do

deallocate(d2sdxdy,dsdy,d2sdx,d2sdy)

!Combining invariant with turbulence coefficients and laplacian
c_turb = 1.0d0*dx
do j = 0,ny-1
  do i = 0,nx-1
    lap_val = (omega_new(i+1,j)+omega_new(i-1,j)-2.0d0*omega_new(i,j))/(dx*dx)
    lap_val = lap_val + (omega_new(i,j+1)+omega_new(i,j-1)-2.0d0*omega_new(i,j))/(dy*dy)
    sgs(i,j) = c_turb*c_turb*sgs(i,j)*lap_val
  end do
end do

deallocate(omega_new,psi_new)

return
end

!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
!Subroutine for smagorinsky based SGS prediction
!Validated
!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
subroutine smag_source_term_less_diss(omega,psi,sgs,nx,ny,dx,dy)

!$ use omp_lib
implicit none
integer, intent(in) :: nx,ny
integer :: i,j
double precision,dimension(0:nx-1,0:ny-1),intent(in) :: omega, psi
double precision, intent(in) :: dx, dy
double precision,dimension(0:nx-1,0:ny-1),intent(out) :: sgs
double precision, dimension(:,:), allocatable :: omega_new, psi_new
double precision, dimension(:,:), allocatable :: dsdy, d2sdx, d2sdy, d2sdxdy
double precision :: c_turb, lap_val

allocate(omega_new(-1:nx,-1:ny))
allocate(psi_new(-1:nx,-1:ny))

do j = 0,ny-1
  do i = 0,nx-1
    omega_new(i,j) = (omega(i,j))
    psi_new(i,j) = psi(i,j)
  end do
end do

!BC Update
call periodic_bc_update(nx,ny,omega_new)
call periodic_bc_update(nx,ny,psi_new)

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

call periodic_bc_update(nx,ny,dsdy)!Need for a second derivative for this quantity

do j = 0,ny-1
  do i = 0,nx-1
    d2sdxdy(i,j) = (dsdy(i+1,j)-dsdy(i-1,j))/(2.0d0*dx)
  end do
end do

!Smag invariant
do j = 0,ny-1
  do i = 0,nx-1
    sgs(i,j) = dsqrt(4.0d0*d2sdxdy(i,j)**2 + (d2sdx(i,j)-d2sdy(i,j))**2)
  end do
end do

deallocate(d2sdxdy,dsdy,d2sdx,d2sdy)

!Combining invariant with turbulence coefficients and laplacian
c_turb = 0.5d0*dx
do j = 0,ny-1
  do i = 0,nx-1
    lap_val = (omega_new(i+1,j)+omega_new(i-1,j)-2.0d0*omega_new(i,j))/(dx*dx)
    lap_val = lap_val + (omega_new(i,j+1)+omega_new(i,j-1)-2.0d0*omega_new(i,j))/(dy*dy)
    sgs(i,j) = c_turb*c_turb*sgs(i,j)*lap_val
  end do
end do

deallocate(omega_new,psi_new)

return
end


!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
!Subroutine for smagorinsky based SGS prediction - using divergence of flux
!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
subroutine smag_source_term_div(omega,psi,sgs,nx,ny,dx,dy)

!$ use omp_lib
implicit none
integer, intent(in) :: nx,ny
integer :: i,j
double precision,dimension(0:nx-1,0:ny-1),intent(in) :: omega, psi
double precision, intent(in) :: dx, dy
double precision,dimension(0:nx-1,0:ny-1),intent(out) :: sgs
double precision, dimension(:,:), allocatable :: omega_new, psi_new
double precision, dimension(:,:), allocatable :: dsdy, d2sdx, d2sdy, d2sdxdy
double precision :: c_turb, lap_val
double precision,dimension(:,:),allocatable :: temp_term, sgs_div

allocate(omega_new(-1:nx,-1:ny))
allocate(psi_new(-1:nx,-1:ny))

do j = 0,ny-1
  do i = 0,nx-1
    omega_new(i,j) = (omega(i,j))
    psi_new(i,j) = psi(i,j)
  end do
end do

!BC Update
call periodic_bc_update(nx,ny,omega_new)
call periodic_bc_update(nx,ny,psi_new)

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

call periodic_bc_update(nx,ny,dsdy)!Need for a second derivative for this quantity

do j = 0,ny-1
  do i = 0,nx-1
    d2sdxdy(i,j) = (dsdy(i+1,j)-dsdy(i-1,j))/(2.0d0*dx)
  end do
end do

!Smag invariant
do j = 0,ny-1
  do i = 0,nx-1
    sgs(i,j) = dsqrt(4.0d0*d2sdxdy(i,j)**2 + (d2sdx(i,j)-d2sdy(i,j))**2)!This is solely EV kernel now
  end do
end do

c_turb = 1.0d0*dx

!Calculating divergence term
allocate(temp_term(-1:nx,-1:ny))
allocate(sgs_div(0:nx-1,0:ny-1))
do j = 0,ny-1
do i = 0,nx-1
  temp_term(i,j) = sgs(i,j)
end do
end do

call periodic_bc_update(nx,ny,temp_term)
!calculating gradient of ev kernel
do j = 0,ny-1
do i = 0,nx-1
  sgs_div(i,j) = (temp_term(i+1,j)-temp_term(i-1,j))*(omega_new(i+1,j)-omega_new(i-1,j))/(4.0*dx*dx)
  sgs_div(i,j) = sgs_div(i,j) + (temp_term(i,j+1)-temp_term(i,j-1))*(omega_new(i,j+1)-omega_new(i,j-1))/(4.0*dy*dy)
  sgs_div(i,j) = c_turb*c_turb*sgs_div(i,j)
end do
end do


deallocate(d2sdxdy,dsdy,d2sdx,d2sdy)

!Combining invariant with turbulence coefficients and laplacian
do j = 0,ny-1
  do i = 0,nx-1
    lap_val = (omega_new(i+1,j)+omega_new(i-1,j)-2.0d0*omega_new(i,j))/(dx*dx)
    lap_val = lap_val + (omega_new(i,j+1)+omega_new(i,j-1)-2.0d0*omega_new(i,j))/(dy*dy)
    sgs(i,j) = c_turb*c_turb*sgs(i,j)*lap_val + sgs_div(i,j)
  end do
end do

deallocate(omega_new,psi_new,temp_term,sgs_div)

return
end


!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
!Subroutine for Leith SGS calculation
!Validated
!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
subroutine leith_source_term(omega,psi,sgs,nx,ny,dx,dy)

!$ use omp_lib
implicit none
integer, intent(in) :: nx,ny
integer :: i,j
double precision,dimension(0:nx-1,0:ny-1),intent(in) :: omega, psi
double precision, intent(in) :: dx, dy
double precision,dimension(0:nx-1,0:ny-1),intent(out) :: sgs
double precision, dimension(:,:), allocatable :: omega_new, psi_new
double precision, dimension(:,:), allocatable :: dwdx,dwdy
double precision :: c_turb, lap_val


allocate(omega_new(-1:nx,-1:ny))
allocate(psi_new(-1:nx,-1:ny))

do j = 0,ny-1
  do i = 0,nx-1
    omega_new(i,j) = (omega(i,j))
    psi_new(i,j) = psi(i,j)
  end do
end do

!BC Update
call periodic_bc_update(nx,ny,omega_new)
call periodic_bc_update(nx,ny,psi_new)

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
    sgs(i,j) = dsqrt(dwdx(i,j)**2 + dwdy(i,j)**2)
  end do
end do

deallocate(dwdx,dwdy)

!Combining invariants with turbulence coefficients and laplacian
c_turb = 1.0d0*dx
do j = 0,ny-1
  do i = 0,nx-1
    lap_val = (omega_new(i+1,j)+omega_new(i-1,j)-2.0d0*omega_new(i,j))/(dx*dx)
    lap_val = lap_val + (omega_new(i,j+1)+omega_new(i,j-1)-2.0d0*omega_new(i,j))/(dy*dy)
    sgs(i,j) = c_turb*c_turb*c_turb*sgs(i,j)*lap_val
  end do
end do

deallocate(omega_new,psi_new)

return
end


!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
!Subroutine for Leith SGS calculation
!Validated
!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
subroutine leith_source_term_div(omega,psi,sgs,nx,ny,dx,dy)

!$ use omp_lib
implicit none
integer, intent(in) :: nx,ny
integer :: i,j
double precision,dimension(0:nx-1,0:ny-1),intent(in) :: omega, psi
double precision, intent(in) :: dx, dy
double precision,dimension(0:nx-1,0:ny-1),intent(out) :: sgs
double precision, dimension(:,:), allocatable :: omega_new, psi_new
double precision, dimension(:,:), allocatable :: dwdx,dwdy
double precision :: c_turb, lap_val
double precision, dimension(:,:), allocatable :: sgs_div, temp_term

allocate(omega_new(-1:nx,-1:ny))
allocate(psi_new(-1:nx,-1:ny))

do j = 0,ny-1
  do i = 0,nx-1
    omega_new(i,j) = (omega(i,j))
    psi_new(i,j) = psi(i,j)
  end do
end do

!BC Update
call periodic_bc_update(nx,ny,omega_new)
call periodic_bc_update(nx,ny,psi_new)

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
    sgs(i,j) = dsqrt(dwdx(i,j)**2 + dwdy(i,j)**2)
  end do
end do

!---------------------------------------------------------------------------------

c_turb = 1.0d0*dx

!Calculating divergence term
allocate(temp_term(-1:nx,-1:ny))
allocate(sgs_div(0:nx-1,0:ny-1))
do j = 0,ny-1
do i = 0,nx-1
  temp_term(i,j) = sgs(i,j)
end do
end do

call periodic_bc_update(nx,ny,temp_term)
!calculating gradient of ev kernel
do j = 0,ny-1
do i = 0,nx-1
  sgs_div(i,j) = (temp_term(i+1,j)-temp_term(i-1,j))*(omega_new(i+1,j)-omega_new(i-1,j))/(4.0*dx*dx)
  sgs_div(i,j) = sgs_div(i,j) + (temp_term(i,j+1)-temp_term(i,j-1))*(omega_new(i,j+1)-omega_new(i,j-1))/(4.0*dy*dy)
  sgs_div(i,j) = c_turb*c_turb*c_turb*sgs_div(i,j)
end do
end do

!Combining invariants with turbulence coefficients and laplacian
do j = 0,ny-1
  do i = 0,nx-1
    lap_val = (omega_new(i+1,j)+omega_new(i-1,j)-2.0d0*omega_new(i,j))/(dx*dx)
    lap_val = lap_val + (omega_new(i,j+1)+omega_new(i,j-1)-2.0d0*omega_new(i,j))/(dy*dy)
    sgs(i,j) = c_turb*c_turb*c_turb*sgs(i,j)*lap_val + sgs_div(i,j)
  end do
end do

!---------------------------------------------------------------------------------

deallocate(dwdx,dwdy)
deallocate(omega_new,psi_new,temp_term,sgs_div)

return
end


!---------------------------------------------------------------------------!
!subroutine - boundary condition update
!Validated
!---------------------------------------------------------------------------!
subroutine periodic_bc_update(nx,ny,u)
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


!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
!Subroutine for Leith kernel calculation
!Validated
!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
subroutine vort_grad(omega,psi,st,nx,ny,dx,dy)

!$ use omp_lib
implicit none
integer :: i,j,nx,ny
double precision,dimension(0:nx-1,0:ny-1),intent(in) :: omega, psi
double precision, intent(in) :: dx, dy
double precision :: st(-2:nx+2,-2:ny+2)
double precision, dimension(:,:), allocatable :: omega_new, psi_new
double precision, dimension(:,:), allocatable :: dwdx,dwdy

allocate(omega_new(-1:nx,-1:ny))
allocate(psi_new(-1:nx,-1:ny))

do j = 0,ny-1
  do i = 0,nx-1
    omega_new(i,j) = (omega(i,j))
    psi_new(i,j) = psi(i,j)
  end do
end do

!BC Update
call periodic_bc_update(nx,ny,omega_new)
call periodic_bc_update(nx,ny,psi_new)

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
    st(i,j) = dsqrt(dwdx(i,j)**2 + dwdy(i,j)**2)
  end do
end do

call periodic_bc_update_dyn(nx,ny,st)

deallocate(dwdx,dwdy)
deallocate(omega_new,psi_new)

return
end


!-------------------------------------------------------------------------------------------!
!Set of routines for dynamic EV SGS calculation
!-------------------------------------------------------------------------------------------!
!-------------------------------------------------------------------------------------------!

!-------------------------------------------------------------------------------------------!
!-------------------------------------------------------------------------------------------!
!Dynamic smagorinsky
!-------------------------------------------------------------------------------------------!
!-------------------------------------------------------------------------------------------!
subroutine dynamic_smagorinsky(omega,psi,laplacian,sgs,nx,ny,dx,dy)
implicit none

!Inputs from python
integer, intent(in) :: nx,ny
double precision,dimension(0:nx-1,0:ny-1),intent(in) :: omega, psi, laplacian
double precision, intent(in) :: dx, dy
double precision,dimension(0:nx-1,0:ny-1),intent(out) :: sgs

integer :: i,j
double precision :: kappa2,dd,nn,csd
double precision, dimension (-2:nx+2,-2:ny+2)  :: jc,vt
double precision, dimension (:,:), allocatable :: omega_f,psi_f,fjc,jcf,st,lwf,omega_new,psi_new,lap_new

kappa2 = 2.0d0

  !compute jacobian of filtered variables
  allocate(omega_new(-2:nx+2,-2:ny+2))
  allocate(psi_new(-2:nx+2,-2:ny+2))
  allocate(lap_new(-2:nx+2,-2:ny+2))

  do j = 0,ny-1
    do i = 0,nx-1
      psi_new(i,j) = psi(i,j)
      omega_new(i,j) = omega(i,j)
      lap_new(i,j) = laplacian(i,j)
    end do
  end do

  call periodic_bc_update_dyn(nx,ny,lap_new)

  allocate(omega_f(-2:nx+2,-2:ny+2))
  allocate(psi_f(-2:nx+2,-2:ny+2))
  allocate(fjc(-2:nx+2,-2:ny+2))

  call filter_dyn(nx,ny,omega_new,omega_f)
  call filter_dyn(nx,ny,psi_new,psi_f)

  call jacobian_arakawa(nx,ny,dx,dy,omega_new,psi_new,jc)
  call jacobian_arakawa(nx,ny,dx,dy,omega_f,psi_f,fjc)

  !compute filtered jacobian
  allocate(jcf(-2:nx+2,-2:ny+2))
  call filter_dyn(nx,ny,jc,jcf)

  !compute laplacian of wf 
  allocate(lwf(-2:nx+2,-2:ny+2))
  call laplacian_calc(nx,ny,dx,dy,omega_f,lwf)

  !compute strain
  allocate(st(-2:nx+2,-2:ny+2))
  call strain(omega,psi,st,nx,ny,dx,dy)
  
  !get filtered st ==> sf
        call filter_dyn(nx,ny,st,psi_f)

  !compute psi_f L on test filter ==> lwf
  do j=0,ny
  do i=0,nx
  lwf(i,j)=psi_f(i,j)*lwf(i,j)
  end do
  end do

  !compute |S|L ==> wf on grid filter
  do j=0,ny
  do i=0,nx
  omega_f(i,j)=st(i,j)*lap_new(i,j)
  end do
  end do

  !compute  filtered |S|L on grid filter
  call filter_dyn(nx,ny,omega_f,psi_f)

  nn = 0.0d0
  dd = 0.0d0
  !compute (cs*delta)^2 =csd
  do j=0,ny
  do i=0,nx 
  nn = nn + (fjc(i,j) - jcf(i,j))*(kappa2*lwf(i,j) - psi_f(i,j))
  dd = dd + (kappa2*lwf(i,j) - psi_f(i,j))*(kappa2*lwf(i,j) - psi_f(i,j))
  end do
  end do
  
  !compute csd
  csd = dabs(nn/dd)


  !Final source term
  do j=0,ny-1
  do i=0,nx-1
    sgs(i,j) = csd*st(i,j)*laplacian(i,j)
  end do
  end do

deallocate(omega_f,psi_f,fjc,jcf,st,lwf)
deallocate(omega_new,psi_new,lap_new)

return
end


!-------------------------------------------------------------------------------------------!
!-------------------------------------------------------------------------------------------!
!Dynamic smagorinsky
!-------------------------------------------------------------------------------------------!
!-------------------------------------------------------------------------------------------!
subroutine dynamic_leith(omega,psi,laplacian,sgs,nx,ny,dx,dy)
implicit none

!Inputs from python
integer, intent(in) :: nx,ny
double precision,dimension(0:nx-1,0:ny-1),intent(in) :: omega, psi, laplacian
double precision, intent(in) :: dx, dy
double precision,dimension(0:nx-1,0:ny-1),intent(out) :: sgs

integer :: i,j
double precision :: kappa2,dd,nn,csd
double precision, dimension (-2:nx+2,-2:ny+2)  :: jc,vt
double precision, dimension (:,:), allocatable :: omega_f,psi_f,fjc,jcf,st,lwf,omega_new,psi_new,lap_new

kappa2 = 2.0d0

  !compute jacobian of filtered variables
  allocate(omega_new(-2:nx+2,-2:ny+2))
  allocate(psi_new(-2:nx+2,-2:ny+2))
  allocate(lap_new(-2:nx+2,-2:ny+2))

  do j = 0,ny-1
    do i = 0,nx-1
      psi_new(i,j) = psi(i,j)
      omega_new(i,j) = omega(i,j)
      lap_new(i,j) = laplacian(i,j)
    end do
  end do

  call periodic_bc_update_dyn(nx,ny,lap_new)

  allocate(omega_f(-2:nx+2,-2:ny+2))
  allocate(psi_f(-2:nx+2,-2:ny+2))
  allocate(fjc(-2:nx+2,-2:ny+2))

  call filter_dyn(nx,ny,omega_new,omega_f)
  call filter_dyn(nx,ny,psi_new,psi_f)

  call jacobian_arakawa(nx,ny,dx,dy,omega_new,psi_new,jc)
  call jacobian_arakawa(nx,ny,dx,dy,omega_f,psi_f,fjc)

  !compute filtered jacobian
  allocate(jcf(-2:nx+2,-2:ny+2))
  call filter_dyn(nx,ny,jc,jcf)

  !compute laplacian of wf 
  allocate(lwf(-2:nx+2,-2:ny+2))
  call laplacian_calc(nx,ny,dx,dy,omega_f,lwf)

  !compute strain
  allocate(st(-2:nx+2,-2:ny+2))
  call vort_grad(omega,psi,st,nx,ny,dx,dy)
  
  !get filtered st ==> sf
        call filter_dyn(nx,ny,st,psi_f)

  !compute psi_f L on test filter ==> lwf
  do j=0,ny
  do i=0,nx
  lwf(i,j)=psi_f(i,j)*lwf(i,j)
  end do
  end do

  !compute |S|L ==> wf on grid filter
  do j=0,ny
  do i=0,nx
  omega_f(i,j)=st(i,j)*lap_new(i,j)
  end do
  end do

  !compute  filtered |S|L on grid filter
  call filter_dyn(nx,ny,omega_f,psi_f)

  nn = 0.0d0
  dd = 0.0d0
  !compute (cs*delta)^2 =csd
  do j=0,ny
  do i=0,nx 
  nn = nn + (fjc(i,j) - jcf(i,j))*(kappa2*lwf(i,j) - psi_f(i,j))
  dd = dd + (kappa2*lwf(i,j) - psi_f(i,j))*(kappa2*lwf(i,j) - psi_f(i,j))
  end do
  end do
  
  !compute csd
  csd = dabs(nn/dd)


  !Final source term
  do j=0,ny-1
  do i=0,nx-1
    sgs(i,j) = csd*st(i,j)*laplacian(i,j)
  end do
  end do

deallocate(omega_f,psi_f,fjc,jcf,st,lwf)
deallocate(omega_new,psi_new,lap_new)

return
end


!---------------------------------------------------------------------------!
!subroutine - boundary condition update
!Validated
!---------------------------------------------------------------------------!
subroutine periodic_bc_update_dyn(nx,ny,u)
implicit none

integer :: nx, ny, i, j
double precision, dimension(-2:nx+2,-2:ny+2) :: u

do i = 0,nx-1
u(i,-1) = u(i,ny-1)
u(i,-2) = u(i,ny-2)
u(i,ny) = u(i,0)
u(i,ny+1) = u(i,1)
u(i,ny+2) = u(i,2)
end do

do j = -2,ny+2
u(-2,j) = u(nx-2,j)
u(-1,j) = u(nx-1,j)
u(nx,j) = u(0,j)
u(nx+1,j) = u(1,j)
u(nx+2,j) = u(2,j)
end do

end subroutine


!-----------------------------------------------------------------!
!Filter
!-----------------------------------------------------------------!
subroutine filter_dyn(nx,ny,w,wf)
implicit none
integer ::nx,ny,ifil,i,j
double precision::w(-2:nx+2,-2:ny+2),wf(-2:nx+2,-2:ny+2),dd

call periodic_bc_update_dyn(nx,ny,w)

dd=1.0d0/16.0d0

do j=0,ny
do i=0,nx
wf(i,j) = dd*(4.0d0*w(i,j) &
       + 2.0d0*(w(i+1,j) + w(i-1,j) + w(i,j+1) + w(i,j-1)) &
     + w(i+1,j+1) + w(i-1,j-1) + w(i+1,j-1) + w(i-1,j+1))
end do
end do

call periodic_bc_update_dyn(nx,ny,wf)

return
end


!-------------------------------------------------------------------------!
! compute jacobian by second order Arakawa scheme (conservative)
! Modified Romit Maulik
!-------------------------------------------------------------------------!
subroutine jacobian_arakawa(nx,ny,dx,dy,w,s,jc)
implicit none
integer :: i,j,nx,ny
double precision :: dx,dy,j1,j2,j3,j11,j22,j33,g,e,z,h
double precision, dimension (-2:nx+2,-2:ny+2)  :: w,s
double precision, dimension (-2:nx+2,-2:ny+2)  :: jc

g = 1.0d0/(4.0d0*dx*dy)
e = 1.0d0/(8.0d0*dx*dy)
z = 2.0d0/3.0d0
h = 1.0d0/3.0d0

call periodic_bc_update_dyn(nx,ny,w)
call periodic_bc_update_dyn(nx,ny,s)

do j = 0,ny
do i = 0,nx


j1 = 1.0/(4.0*dx*dy) * ((w(i+1,j)-w(i-1,j)) * (s(i,j+1) - s(i,j-1)) &
      - (w(i,j+1)-w(i,j-1)) * (s(i+1,j) - s(i-1,j)))

j2 = 1.0 / (4.0 * dx * dy) * (w(i+1, j) * (s(i+1, j+1) - s(i+1, j-1)) &
                                         - w(i-1, j) * (s(i-1, j+1) - s(i-1, j-1)) &
                                         - w(i, j+1) * (s(i+1, j+1) - s(i-1, j+1)) &
                                         + w(i, j-1) * (s(i+1, j-1) - s(i-1, j-1)) &
                                          )

j3 = 1.0 / (4.0 * dx * dy) * (w(i+1, j+1) * (s(i, j+1) - s(i+1, j)) &
                                        -  w(i-1, j-1) * (s(i-1, j) - s(i, j-1)) &
                                        -  w(i-1, j+1) * (s(i, j+1) - s(i-1, j)) &
                                        +  w(i+1, j-1) * (s(i+1, j) - s(i, j-1)) &
                                          )

jc(i, j) = (j1 + j2 + j3)/3.0

end do
end do

call periodic_bc_update_dyn(nx,ny,jc)

return
end


!-------------------------------------------------------------------------!
! compute laplacian
! Modified second order - Romit 
!-------------------------------------------------------------------------!
subroutine laplacian_calc(nx,ny,dx,dy,u,lp)
implicit none
integer :: i,j,nx,ny
double precision :: dx,dy, d2wdy2, d2wdx2
double precision, dimension (-2:nx+2,-2:ny+2)  :: u,lp
double precision, dimension (:), allocatable   :: a,b

call periodic_bc_update_dyn(nx,ny,u) 

do j = 0,ny
  do i = 0,nx
    d2wdy2 = (u(i, j+1) + u(i, j-1) - 2.0 * u(i, j)) / (dy * dy)
    d2wdx2 = (u(i+1, j) + u(i-1, j) - 2.0 * u(i, j)) / (dx * dx)

    lp(i,j) = d2wdx2 + d2wdy2

  end do
end do

call periodic_bc_update_dyn(nx,ny,lp)

return
end


!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
!Subroutine for smagorinsky kernel calculation
!Validated
!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
subroutine strain(omega,psi,st,nx,ny,dx,dy)

!$ use omp_lib
implicit none
integer:: nx,ny,i,j
double precision,dimension(0:nx-1,0:ny-1),intent(in) :: omega, psi
double precision, intent(in) :: dx, dy
double precision :: st(-2:nx+2,-2:ny+2)
double precision, dimension(:,:), allocatable :: omega_new, psi_new
double precision, dimension(:,:), allocatable :: dsdy, d2sdx, d2sdy, d2sdxdy

allocate(omega_new(-1:nx,-1:ny))
allocate(psi_new(-1:nx,-1:ny))

do j = 0,ny-1
  do i = 0,nx-1
    omega_new(i,j) = (omega(i,j))
    psi_new(i,j) = psi(i,j)
  end do
end do

!BC Update
call periodic_bc_update(nx,ny,omega_new)
call periodic_bc_update(nx,ny,psi_new)

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

call periodic_bc_update(nx,ny,dsdy)!Need for a second derivative for this quantity

do j = 0,ny-1
  do i = 0,nx-1
    d2sdxdy(i,j) = (dsdy(i+1,j)-dsdy(i-1,j))/(2.0d0*dx)
  end do
end do

!Smag invariant
do j = 0,ny-1
  do i = 0,nx-1
    st(i,j) = dsqrt(4.0d0*d2sdxdy(i,j)**2 + (d2sdx(i,j)-d2sdy(i,j))**2)
  end do
end do

call periodic_bc_update_dyn(nx,ny,st)

deallocate(d2sdxdy,dsdy,d2sdx,d2sdy)
deallocate(omega_new,psi_new)

return
end


!-----------------------------------------------------------------!
!Trapezoidal filter
!-----------------------------------------------------------------!
subroutine filter_trapezoidal(q_org,nx,ny)
implicit none
integer,intent(in)::nx,ny
double precision, intent(inout) :: q_org(0:nx-1,0:ny-1)
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
