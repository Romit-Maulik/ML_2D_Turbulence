
!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
!Subroutine for sampling across domain for SGS predictions using ML
!Used for Feature-SGS prediction//No stencil, only pointwise
!Validated
!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
subroutine field_sampler(omega,psi,sampling_matrix,nx,ny,dx,dy)

!$ use omp_lib
implicit none
integer, intent(in) :: nx,ny
integer :: i,j, sample
double precision,dimension(0:nx-1,0:ny-1),intent(in) :: omega, psi
double precision, intent(in) :: dx, dy
double precision,dimension(0:nx*ny,0:8),intent(out) :: sampling_matrix
double precision, dimension(:,:), allocatable :: omega_new, psi_new, lapc
double precision, dimension(:,:), allocatable :: dsdy, d2sdx, d2sdy, d2sdxdy,dwdx,dwdy
double precision,dimension(:,:,:),allocatable :: inv
double precision :: inv1_mean, inv2_mean, inv3_mean, inv4_mean, lap_val
double precision :: inv1_std, inv2_std, inv3_std, inv4_std
double precision, dimension(:,:),allocatable :: s11, s12, s22, r12


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

allocate(inv(0:nx,0:ny,0:3))
!Smag invariant
do j = 0,ny-1
  do i = 0,nx-1
    inv(i,j,0) = dsqrt(4.0d0*d2sdxdy(i,j)**2 + (d2sdx(i,j)-d2sdy(i,j))**2)
  end do
end do



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
    inv(i,j,1) = dsqrt(dwdx(i,j)**2 + dwdy(i,j)**2)
  end do
end do

!BL invariant
do j = 0,ny-1
  do i = 0,nx-1
    inv(i,j,2) = dabs(omega_new(i,j))
  end do
end do

!CS invariant
do j = 0,ny-1
  do i = 0,nx-1
    inv(i,j,3) = dsqrt(d2sdx(i,j)**2 + d2sdy(i,j)**2)
  end do
end do


deallocate(dwdx,dwdy)
deallocate(d2sdxdy,dsdy,d2sdx,d2sdy)

!Calculating means for input vectors
inv1_mean = 0.0d0
inv2_mean = 0.0d0
inv3_mean = 0.0d0
inv4_mean = 0.0d0

do j = 0,ny-1
  do i = 0,nx-1
  	inv1_mean = inv1_mean + inv(i,j,0)
  	inv2_mean = inv2_mean + inv(i,j,1)
	inv3_mean = inv3_mean + inv(i,j,2)
  	inv4_mean = inv4_mean + inv(i,j,3)
  end do
end do


inv1_mean = inv1_mean/dfloat(nx*ny)
inv2_mean = inv2_mean/dfloat(nx*ny)
inv3_mean = inv3_mean/dfloat(nx*ny)
inv4_mean = inv4_mean/dfloat(nx*ny)


!Calculating sdevs for input vectors

inv1_std = 0.0d0
inv2_std = 0.0d0
inv3_std = 0.0d0
inv4_std = 0.0d0

do j = 0,ny-1
  do i = 0,nx-1
  	inv1_std = inv1_std + (inv(i,j,0)-inv1_mean)**2
  	inv2_std = inv2_std + (inv(i,j,1)-inv2_mean)**2
  	inv3_std = inv3_std + (inv(i,j,2)-inv3_mean)**2
  	inv4_std = inv4_std + (inv(i,j,3)-inv4_mean)**2
  end do
end do


inv1_std = dsqrt(inv1_std/dfloat(nx*ny-1))
inv2_std = dsqrt(inv2_std/dfloat(nx*ny-1))
inv3_std = dsqrt(inv3_std/dfloat(nx*ny-1))
inv4_std = dsqrt(inv4_std/dfloat(nx*ny-1))


!Changing input data arrays - normalization
do j = 0,ny-1
  do i = 0,nx-1
	inv(i,j,0) = (inv(i,j,0) - inv1_mean)/inv1_std
	inv(i,j,1) = (inv(i,j,1) - inv2_mean)/inv2_std
	inv(i,j,2) = (inv(i,j,2) - inv3_mean)/inv3_std
	inv(i,j,3) = (inv(i,j,3) - inv4_mean)/inv4_std
  end do
end do


!Finding field features
allocate(s11(0:nx-1,0:ny-1))
allocate(s12(0:nx-1,0:ny-1))
allocate(s22(0:nx-1,0:ny-1))
allocate(r12(0:nx-1,0:ny-1))

call strain_rotation_tensor_calc(nx,ny,psi,s11,s12,s22,r12,dx,dy)

!Find laplacian
allocate(lapc(0:nx-1,0:ny-1))

!Calculating laplacian
do j = 0,ny-1
  do i = 0,nx-1
    lap_val = (omega_new(i+1,j)+omega_new(i-1,j)-2.0d0*omega_new(i,j))/(dx*dx)
    lapc(i,j) = lap_val + (omega_new(i,j+1)+omega_new(i,j-1)-2.0d0*omega_new(i,j))/(dy*dy)
  end do
end do

!Normalizing Laplacian to zero mean and unit variance
!Calculating means for laplacian - reusing inv1_mean, inv1_std
inv1_mean = 0.0d0
do j = 0,ny-1
  do i = 0,nx-1
  	inv1_mean = inv1_mean + lapc(i,j)
  end do
end do
inv1_mean = inv1_mean/dfloat(nx*ny)
!Calculating sdevs for input vectors
inv1_std = 0.0d0
do j = 0,ny-1
  do i = 0,nx-1
  	inv1_std = inv1_std + (lapc(i,j)-inv1_mean)**2
  end do
end do

inv1_std = dsqrt(inv1_std/dfloat(nx*ny-1))
!Changing input data arrays - normalization
do j = 0,ny-1
  do i = 0,nx-1
	lapc(i,j) = (lapc(i,j) - inv1_mean)/inv1_std
  end do
end do


sample = 0
!Preparing sample matrix
!$OMP PARALLEL DO
do j = 0,ny-1
  do i = 0,nx-1

    sampling_matrix(sample,0) = s11(i,j)
    sampling_matrix(sample,1) = s12(i,j)
    sampling_matrix(sample,2) = s22(i,j)
    sampling_matrix(sample,3) = r12(i,j)

    sampling_matrix(sample,4) = inv(i,j,0)
    sampling_matrix(sample,5) = inv(i,j,1)
    sampling_matrix(sample,6) = inv(i,j,2)
    sampling_matrix(sample,7) = inv(i,j,3)

    sampling_matrix(sample,8) = lapc(i,j)

    sample = sample + 1

  end do
end do
!$OMP END PARALLEL DO

deallocate(omega_new,psi_new,inv)
deallocate(s11,s12,s22,r12,lapc)

return
end


!--------------------------------------------------------------------------
!Subroutine for reshaping Keras prediction into nx x ny shape
!The return matrix here contains values for source term (Pi from Keras)
!Validated
!--------------------------------------------------------------------------

subroutine sgs_reshape(return_matrix,nrows,ncols,sgs,laplacian,nx,ny)

!$ use omp_lib
implicit none

integer, intent(in) :: nx,ny,nrows,ncols
integer :: i,j, sample
double precision, dimension(0:nrows,ncols),intent(in) :: return_matrix
double precision, dimension(0:nx-1,0:ny-1),intent(inout) :: sgs
double precision, dimension(0:nx-1,0:ny-1),intent(in) :: laplacian

sample = 0
!Reading return matrix
!$OMP PARALLEL DO
do j = 0,ny-1
  do i = 0,nx-1
  !This if statement to ensure no negative numerical viscosities
  !Key for JFM Rapids manuscript
    if (laplacian(i,j)<0 .and. return_matrix(sample,1)<0) then
      sgs(i,j) = return_matrix(sample,1)
    else if (laplacian(i,j)>0 .and. return_matrix(sample,1)>0) then
      sgs(i,j) = return_matrix(sample,1)
    else
      sgs(i,j) = 0.0d0
    end if

    sample = sample + 1
  
  end do
end do
!$OMP END PARALLEL DO


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
!Subroutine for reshaping Keras prediction into nx x ny shape
!The return matrix here contains values for source term (Pi from Keras)
!A form of local averaging to statistically represent backscatter in a stencil
!--------------------------------------------------------------------------

subroutine sgs_reshape_backscatter(return_matrix,nrows,ncols,sgs,laplacian,nx,ny)

!$ use omp_lib
implicit none

integer, intent(in) :: nx,ny,nrows,ncols
integer :: i,j, sample, k
double precision, dimension(0:nrows,ncols),intent(in) :: return_matrix
double precision, dimension(0:nx-1,0:ny-1),intent(inout) :: sgs
double precision, dimension(0:nx-1,0:ny-1),intent(in) :: laplacian


!Temp variables in Fortran
double precision, dimension(:,:), allocatable :: sgs_temp, lap_temp
double precision, dimension(1:9) :: nu, lap
double precision :: sgs_val, lap_val, nu_av, tiny_val, lap_av


tiny_val = 1.0d-10

allocate(sgs_temp(-1:nx,-1:ny))
allocate(lap_temp(-1:nx,-1:ny))

sample = 0
!Reading return matrix
!$OMP PARALLEL DO
do j = 0,ny-1
  do i = 0,nx-1
  sgs_temp(i,j) = return_matrix(sample,1)
  lap_temp(i,j) = laplacian(i,j)
  sample = sample + 1
  end do
end do
!$OMP END PARALLEL DO

!Boundary conditions update
!BC Update
call periodic_bc_update(nx,ny,sgs_temp)
call periodic_bc_update(nx,ny,lap_temp)

!Local averaging
!$OMP PARALLEL DO
do j = 0,ny-1
  do i = 0,nx-1

  !Calculate eddy viscosities
  sgs_val = sgs_temp(i,j)
  lap_val = lap_temp(i,j)
  nu(1) = sgs_val/lap_val
  lap(1) = lap_val

  sgs_val = sgs_temp(i,j+1)
  lap_val = lap_temp(i,j+1)
  nu(2) = sgs_val/lap_val
  lap(2) = lap_val

  sgs_val = sgs_temp(i,j-1)
  lap_val = lap_temp(i,j-1)
  nu(3) = sgs_val/lap_val
  lap(3) = lap_val

  sgs_val = sgs_temp(i+1,j)
  lap_val = lap_temp(i+1,j)
  nu(4) = sgs_val/lap_val
  lap(4) = lap_val

  sgs_val = sgs_temp(i+1,j+1)
  lap_val = lap_temp(i+1,j+1)
  nu(5) = sgs_val/lap_val
  lap(5) = lap_val

  sgs_val = sgs_temp(i+1,j-1)
  lap_val = lap_temp(i+1,j-1)
  nu(6) = sgs_val/lap_val
  lap(6) = lap_val

  sgs_val = sgs_temp(i-1,j)
  lap_val = lap_temp(i-1,j)
  nu(7) = sgs_val/lap_val
  lap(7) = lap_val

  sgs_val = sgs_temp(i-1,j+1)
  lap_val = lap_temp(i-1,j+1)
  nu(8) = sgs_val/lap_val
  lap(8) = lap_val

  sgs_val = sgs_temp(i-1,j-1)
  lap_val = lap_temp(i-1,j-1)
  nu(9) = sgs_val/lap_val
  lap(9) = lap_val

  nu_av = 0.0d0
  lap_av = 0.0d0
  do k = 1,9
      nu_av = nu_av + nu(k)
      lap_av = lap_av + lap(k)
  end do
  nu_av = nu_av/9.0d0
  lap_av = lap_av/9.0d0


  if (nu(1)>tiny_val.and.nu_av>nu(1)) then
           sgs(i,j) = nu(1)*laplacian(i,j)
  else
     sgs(i,j) = 0.0d0
  end if

 
  end do
end do
!$OMP END PARALLEL DO

deallocate(sgs_temp,lap_temp)


return
end


!------------------------------------------------------------------------------!
!------------------------------------------------------------------------------!
!subroutine to calculate strain and rotation tensors
!Outputs normalized to zero mean and unit variance
!------------------------------------------------------------------------------!
!------------------------------------------------------------------------------!
subroutine strain_rotation_tensor_calc(nxc,nyc,sc,s11,s12,s22,r12,dx,dy)
  implicit none

  integer :: nxc,nyc, i, j
  double precision, dimension(0:nxc-1,0:nyc-1) :: sc,s11,s12,s22,r12
  double precision :: dx, dy

  double precision, dimension(:,:), allocatable :: u, v, sc_temp
  double precision :: s11_m, s12_m, s22_m, r12_m
  double precision :: s11_s, s12_s, s22_s, r12_s

  allocate(u(-1:nxc,-1:nyc))
  allocate(v(-1:nxc,-1:nyc))
  allocate(sc_temp(-1:nxc,-1:nyc))

  do j=0,nyc-1
    do i = 0,nxc-1
      sc_temp(i,j)=sc(i,j)
    end do
  end do

  call periodic_bc_update(nxc,nyc,sc_temp)

  !Calculate u and v
  do j=0,nyc-1
    do i = 0,nxc-1
      u(i,j)=(sc_temp(i,j+1)-sc_temp(i,j-1))/(2.0d0*dy)
      v(i,j)=-(sc_temp(i+1,j)-sc_temp(i-1,j))/(2.0d0*dx)
    end do
  end do

  call periodic_bc_update(nxc,nyc,u)
  call periodic_bc_update(nxc,nyc,v)

  !Calculate strain and rotation components - 2d
  do j=0,nyc-1
    do i = 0,nxc-1
      s11(i,j) = (u(i+1,j)-u(i-1,j))/(2.0d0*dx)
      s22(i,j) = (v(i,j+1)-v(i,j-1))/(2.0d0*dy)
      s12(i,j) = 0.5d0*((u(i,j+1)-u(i,j-1))/(2.0d0*dy) + (v(i+1,j)-v(i-1,j))/(2.0d0*dx))
      r12(i,j) = 0.5d0*((u(i,j+1)-u(i,j-1))/(2.0d0*dy) - (v(i+1,j)-v(i-1,j))/(2.0d0*dx))
    end do
  end do

  !Normalize for deployment
  !Calculating means for input vectors
  s11_m = 0.0d0
  s12_m = 0.0d0
  s22_m = 0.0d0
  r12_m = 0.0d0

  do j = 0,nyc-1
    do i = 0,nxc-1
      s11_m = s11_m + s11(i,j)
      s12_m = s12_m + s12(i,j)
      s22_m = s22_m + s22(i,j)
      r12_m = r12_m + r12(i,j)
    end do
  end do


  s11_m = s11_m/dfloat(nxc*nyc)
  s12_m = s12_m/dfloat(nxc*nyc)
  s22_m = s22_m/dfloat(nxc*nyc)
  r12_m = r12_m/dfloat(nxc*nyc)

  !Calculating sdevs for input vectors

  s11_s = 0.0d0
  s12_s = 0.0d0
  s22_s = 0.0d0
  r12_s = 0.0d0

  do j = 0,nyc-1
    do i = 0,nxc-1
      s11_s = s11_s + (s11(i,j)-s11_m)**2
      s12_s = s12_s + (s12(i,j)-s12_m)**2
      s22_s = s22_s + (s22(i,j)-s22_m)**2
      r12_s = r12_s + (r12(i,j)-r12_m)**2
    end do
  end do


  s11_s = dsqrt(s11_s/dfloat(nxc*nyc-1))
  s12_s = dsqrt(s12_s/dfloat(nxc*nyc-1))
  s22_s = dsqrt(s22_s/dfloat(nxc*nyc-1))
  r12_s = dsqrt(r12_s/dfloat(nxc*nyc-1))

  !Changing input data arrays - normalization
  do j = 0,nyc-1
    do i = 0,nxc-1
    s11(i,j) = (s11(i,j) - s11_m)/s11_s
    s12(i,j) = (s12(i,j) - s12_m)/s12_s
    s22(i,j) = (s22(i,j) - s22_m)/s22_s
    r12(i,j) = (r12(i,j) - r12_m)/r12_s
    end do
  end do


  deallocate(u,v,sc_temp)

  return
end