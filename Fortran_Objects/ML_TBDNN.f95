!--------------------------------------------------------------------------
!Subroutine for reshaping Keras prediction into nx x ny shape
!The return matrix here contains values for source term (Pi from Keras)
!Validated
!--------------------------------------------------------------------------

subroutine sgs_reshape(return_matrix,nrows,ncols,sgs,nx,ny)

!$ use omp_lib
implicit none

integer, intent(in) :: nx,ny,nrows,ncols
integer :: i,j, sample
double precision, dimension(0:nrows,ncols),intent(in) :: return_matrix
double precision, dimension(0:nx-1,0:ny-1),intent(inout) :: sgs

sample = 0
!Reading return matrix
do j = 0,ny-1
  do i = 0,nx-1
      sgs(i,j) = return_matrix(sample,1)
      sample = sample + 1
  end do
end do


return
end

!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
!Subroutine for sampling across domain for SGS predictions using ML
!Used for ML-SGS prediction - input preparation for JFM Rapids
!Validated
!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
subroutine field_sampler(omega,psi,sampling_matrix,nx,ny)

!$ use omp_lib
implicit none
integer, intent(in) :: nx,ny
integer :: i,j, sample
double precision,dimension(0:nx-1,0:ny-1),intent(in) :: omega, psi
double precision,dimension(0:nx*ny,0:17),intent(out) :: sampling_matrix
double precision, dimension(:,:), allocatable :: omega_new, psi_new


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


sample = 0
!Preparing sample matrix
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

    sample = sample + 1

  end do
end do

deallocate(omega_new,psi_new)

return
end


!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
!Subroutine for basis matrix maker for projection of TBDNN
!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
subroutine basis_matrix_shaper(sgs_smag,sgs_leith,sfs_ad,basis_matrix,nx,ny)

!$ use omp_lib
implicit none
integer, intent(in) :: nx,ny
integer :: i,j, sample
double precision,dimension(0:nx-1,0:ny-1),intent(in) :: sgs_smag, sgs_leith, sfs_ad
double precision,dimension(0:nx*ny,3),intent(out) :: basis_matrix


sample = 0
!Preparing sample matrix
do j = 0,ny-1
  do i = 0,nx-1
    basis_matrix(sample,1) = sgs_smag(i,j)
    basis_matrix(sample,2) = sgs_leith(i,j)
    basis_matrix(sample,3) = sfs_ad(i,j)
    
    sample = sample + 1
  end do
end do

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
