!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
!Subroutine for sampling across domain for SGS predictions using ML
!Used for ML-SGS prediction - input preparation for JFM Rapids
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
double precision,dimension(0:nx*ny,0:19),intent(out) :: sampling_matrix
double precision, dimension(:,:), allocatable :: omega_new, psi_new
double precision, dimension(:,:), allocatable :: dsdy, d2sdx, d2sdy, d2sdxdy,dwdx,dwdy
double precision,dimension(:,:,:),allocatable :: inv
!double precision :: vort_mean, stream_mean, inv1_mean, inv2_mean
!double precision :: vort_std, stream_std, inv1_std, inv2_std


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

allocate(inv(0:nx,0:ny,0:1))
!Smag invariant
do j = 0,ny-1
  do i = 0,nx-1
    inv(i,j,0) = dsqrt(4.0d0*d2sdxdy(i,j)**2 + (d2sdx(i,j)-d2sdy(i,j))**2)
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
    inv(i,j,1) = dsqrt(dwdx(i,j)**2 + dwdy(i,j)**2)
  end do
end do

deallocate(dwdx,dwdy)

!!Calculating means for input vectors
!vort_mean = 0.0d0
!stream_mean = 0.0d0
!inv1_mean = 0.0d0
!inv2_mean = 0.0d0

!do j = 0,ny-1
!  do i = 0,nx-1
!  	vort_mean = vort_mean + omega_new(i,j)
!  	stream_mean = stream_mean + psi_new(i,j)
!  	inv1_mean = inv1_mean + inv(i,j,0)
!  	inv2_mean = inv2_mean + inv(i,j,1)
!  end do
!end do

!vort_mean = vort_mean/dfloat(nx*ny)
!stream_mean = stream_mean/dfloat(nx*ny)
!inv1_mean = inv1_mean/dfloat(nx*ny)
!inv2_mean = inv2_mean/dfloat(nx*ny)

!!Calculating sdevs for input vectors
!vort_std = 0.0d0
!stream_std = 0.0d0
!inv1_std = 0.0d0
!inv2_std = 0.0d0

!do j = 0,ny-1
!  do i = 0,nx-1
!  	vort_std = vort_std + (omega_new(i,j)-vort_mean)**2
!  	stream_std = stream_std + (psi_new(i,j)-stream_mean)**2
!  	inv1_std = inv1_std + (inv(i,j,0)-inv1_mean)**2
!  	inv2_std = inv2_std + (inv(i,j,1)-inv2_mean)**2
!  end do
!end do

!vort_std = dsqrt(vort_std/dfloat(nx*ny-1))
!stream_std = dsqrt(stream_std/dfloat(nx*ny-1))
!inv1_std = dsqrt(inv1_std/dfloat(nx*ny-1))
!inv2_std = dsqrt(inv2_std/dfloat(nx*ny-1))


!!Changing input data arrays - normalization
!do j = 0,ny-1
!  do i = 0,nx-1
!  	omega_new(i,j) = (omega_new(i,j) - vort_mean)/vort_std
!	psi_new(i,j) = (psi_new(i,j) - stream_mean)/stream_std
!	inv(i,j,0) = (inv(i,j,0) - inv1_mean)/inv1_std
!	inv(i,j,1) = (inv(i,j,1) - inv2_mean)/inv2_std
!  end do
!end do

sample = 0
!Preparing sample matrix
!$OMP PARALLEL DO
do j = 0,ny-1
  do i = 0,nx-1
    sampling_matrix(sample,0) = omega_new(i,j)
    sampling_matrix(sample,1) = omega_new(i,j+1)
    sampling_matrix(sample,2) = omega_new(i,j-1)
    sampling_matrix(sample,3) = omega_new(i+1,j)
    sampling_matrix(sample,4) = omega_new(i+1,j+1)
    sampling_matrix(sample,5) = omega_new(i+1,j-1)
    sampling_matrix(sample,6) = omega_new(i-1,j)
    sampling_matrix(sample,7) = omega_new(i-1,j+1)
    sampling_matrix(sample,8) = omega_new(i-1,j-1)

    sampling_matrix(sample,9) = psi_new(i,j)
    sampling_matrix(sample,10) = psi_new(i,j+1)
    sampling_matrix(sample,11) = psi_new(i,j-1)
    sampling_matrix(sample,12) = psi_new(i+1,j)
    sampling_matrix(sample,13) = psi_new(i+1,j+1)
    sampling_matrix(sample,14) = psi_new(i+1,j-1)
    sampling_matrix(sample,15) = psi_new(i-1,j)
    sampling_matrix(sample,16) = psi_new(i-1,j+1)
    sampling_matrix(sample,17) = psi_new(i-1,j-1)

    sampling_matrix(sample,18) = inv(i,j,0)
    sampling_matrix(sample,19) = inv(i,j,1)

    sample = sample + 1

  end do
end do
!$OMP END PARALLEL DO

deallocate(omega_new,psi_new,inv)

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

