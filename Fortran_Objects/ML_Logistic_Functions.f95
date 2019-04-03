!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
!Subroutine for sampling across domain for SGS predictions using ML
!Used for ML-Logistic preparation
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

    sample = sample + 1

  end do
end do
!$OMP END PARALLEL DO

deallocate(omega_new,psi_new)

return
end


!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
!Subroutine for sampling across domain for SGS predictions using ML  (vorticity only)
!Used for ML-Logistic preparation
!Validated
!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
subroutine field_sampler_f(f,sampling_matrix,nx,ny)

!$ use omp_lib
implicit none
integer, intent(in) :: nx,ny
integer :: i,j, sample
double precision,dimension(0:nx-1,0:ny-1),intent(in) :: f
double precision,dimension(0:nx*ny,0:8),intent(out) :: sampling_matrix
double precision, dimension(:,:), allocatable :: f_new

allocate(f_new(-1:nx,-1:ny))

do j = 0,ny-1
  do i = 0,nx-1
    f_new(i,j) = (f(i,j))
  end do
end do

!BC Update
call periodic_bc_update(nx,ny,f_new)

sample = 0
!Preparing sample matrix
!$OMP PARALLEL DO
do j = 0,ny-1
  do i = 0,nx-1
    sampling_matrix(sample,0) = f_new(i,j)
    sampling_matrix(sample,1) = f_new(i,j+1)
    sampling_matrix(sample,2) = f_new(i,j-1)
    sampling_matrix(sample,3) = f_new(i+1,j)
    sampling_matrix(sample,4) = f_new(i+1,j+1)
    sampling_matrix(sample,5) = f_new(i+1,j-1)
    sampling_matrix(sample,6) = f_new(i-1,j)
    sampling_matrix(sample,7) = f_new(i-1,j+1)
    sampling_matrix(sample,8) = f_new(i-1,j-1)

    sample = sample + 1

  end do
end do
!$OMP END PARALLEL DO

deallocate(f_new)

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
!Classify the choice of model according to logistic output
!--------------------------------------------------------------------------

subroutine sgs_calculate(return_matrix,nrows,ncols,sgs,sgs_smag,sgs_ad,nx,ny)

!$ use omp_lib
implicit none

integer, intent(in) :: nx,ny,nrows,ncols
integer :: i,j
double precision, dimension(0:nrows,ncols),intent(in) :: return_matrix
double precision, dimension(0:nx-1,0:ny-1),intent(inout) :: sgs
double precision, dimension(0:nx-1,0:ny-1),intent(in) :: sgs_smag, sgs_ad

integer, dimension(:,:), allocatable :: label_field

allocate(label_field(0:nx-1,0:ny-1))
call one_hot_reshape(return_matrix,nrows,ncols,label_field,nx,ny)

do j = 0,ny-1
  do i = 0,nx-1
      if (label_field(i,j) == 1) then
        sgs(i,j) = sgs_ad(i,j)
      else if (label_field(i,j) == 2) then
        sgs(i,j) = sgs_smag(i,j)
	  else if (label_field(i,j) == 3) then
	    sgs(i,j) = 0.0d0
      end if
  end do
end do

deallocate(label_field)

return
end


!--------------------------------------------------------------------------
!Classify the choice of model according to logistic output
!--------------------------------------------------------------------------

subroutine sgs_calculate_blended(return_matrix,nrows,ncols,sgs,sgs_smag,sgs_ad,nx,ny)

!$ use omp_lib
implicit none

integer, intent(in) :: nx,ny,nrows,ncols
integer :: i,j
double precision, dimension(0:nrows,ncols),intent(in) :: return_matrix
double precision, dimension(0:nx-1,0:ny-1),intent(inout) :: sgs
double precision, dimension(0:nx-1,0:ny-1),intent(in) :: sgs_smag, sgs_ad

double precision, dimension(:,:,:), allocatable :: label_field

allocate(label_field(0:nx-1,0:ny-1,3))
call one_hot_reshape_blended(return_matrix,nrows,ncols,label_field,nx,ny)

do j = 0,ny-1
  do i = 0,nx-1
      sgs(i,j) = label_field(i,j,1)*sgs_ad(i,j) + label_field(i,j,2)*sgs_smag(i,j)
  end do
end do

deallocate(label_field)

return
end


!--------------------------------------------------------------------------
!Classify the choice of model according to logistic output - five classes
!--------------------------------------------------------------------------

subroutine sgs_calculate_blended_five_class(return_matrix,nrows,ncols,sgs, sgs_leith, sgs_smag, sfs_ad, sfs_bd ,nx,ny)

!$ use omp_lib
implicit none

integer, intent(in) :: nx,ny,nrows,ncols
integer :: i,j
double precision, dimension(0:nrows,ncols),intent(in) :: return_matrix
double precision, dimension(0:nx-1,0:ny-1),intent(inout) :: sgs
double precision, dimension(0:nx-1,0:ny-1),intent(in) :: sgs_leith, sfs_ad, sfs_bd, sgs_smag

double precision, dimension(:,:,:), allocatable :: label_field

allocate(label_field(0:nx-1,0:ny-1,5))
call one_hot_reshape_blended_five_class(return_matrix,nrows,ncols,label_field,nx,ny)

do j = 0,ny-1
  do i = 0,nx-1
      sgs(i,j) = label_field(i,j,1)*sfs_bd(i,j) + label_field(i,j,2)*sfs_ad(i,j) &
                 & + label_field(i,j,3)*sgs_leith(i,j) + label_field(i,j,4)*sgs_smag(i,j)
  end do
end do

deallocate(label_field)

return
end


!--------------------------------------------------------------------------
!Subroutine for classifying models according to softmax classification
!The return matrix here contains values for labels (one-hot)
!--------------------------------------------------------------------------

subroutine one_hot_reshape(return_matrix,nrows,ncols,label_field,nx,ny)
implicit none

integer :: nx,ny,nrows,ncols
integer :: i,j, sample
double precision, dimension(0:nrows,ncols) :: return_matrix
integer, dimension(0:nx-1,0:ny-1) :: label_field


sample = 0
!Reading return matrix
!$OMP PARALLEL DO
do j = 0,ny-1
  do i = 0,nx-1

    if (return_matrix(sample,1)>return_matrix(sample,2).and.return_matrix(sample,1)>return_matrix(sample,3)) then
        label_field(i,j) = 1      !AD prediction
    else if (return_matrix(sample,2)>return_matrix(sample,1).and.return_matrix(sample,2)>return_matrix(sample,3)) then
        label_field(i,j) = 2      !Smag prediction
    else
        label_field(i,j) = 3      !No model prediction
    end if

    sample = sample + 1 
  end do
end do
!$OMP END PARALLEL DO


return
end



!--------------------------------------------------------------------------
!Subroutine for blending models according to softmax output
!--------------------------------------------------------------------------

subroutine one_hot_reshape_blended(return_matrix,nrows,ncols,label_field,nx,ny)
implicit none

integer :: nx,ny,nrows,ncols
integer :: i,j, sample
double precision, dimension(0:nrows,ncols) :: return_matrix
double precision, dimension(0:nx-1,0:ny-1,3) :: label_field


sample = 0
!Reading return matrix
!$OMP PARALLEL DO
do j = 0,ny-1
  do i = 0,nx-1

    label_field(i,j,1) = return_matrix(sample,1)
    label_field(i,j,2) = return_matrix(sample,2)
    label_field(i,j,3) = return_matrix(sample,3)

    sample = sample + 1 
  end do
end do
!$OMP END PARALLEL DO


return
end

!--------------------------------------------------------------------------
!Subroutine for blending models according to softmax output - five classes
!--------------------------------------------------------------------------

subroutine one_hot_reshape_blended_five_class(return_matrix,nrows,ncols,label_field,nx,ny)
implicit none

integer :: nx,ny,nrows,ncols
integer :: i,j, sample
double precision, dimension(0:nrows,ncols) :: return_matrix
double precision, dimension(0:nx-1,0:ny-1,5) :: label_field


sample = 0
!Reading return matrix
!$OMP PARALLEL DO
do j = 0,ny-1
  do i = 0,nx-1

    label_field(i,j,1) = return_matrix(sample,1)
    label_field(i,j,2) = return_matrix(sample,2)
    label_field(i,j,3) = return_matrix(sample,3)
    label_field(i,j,4) = return_matrix(sample,4)
    label_field(i,j,5) = return_matrix(sample,5)

    sample = sample + 1 
  end do
end do
!$OMP END PARALLEL DO


return
end
