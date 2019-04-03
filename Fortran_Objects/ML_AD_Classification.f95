!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
!subroutine for making a random sampling matrix from omega field
!Used for ML-AD classification
!Validated
!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
subroutine field_sampler(omega,nx,ny,sampling_matrix,n_samples)

!$ use omp_lib
implicit none

integer, intent(in) :: nx,ny,n_samples
integer :: i,j,sample,seed
double precision,dimension(0:nx-1,0:ny-1),intent(in) :: omega
double precision,dimension(0:n_samples-1,0:8),intent(out) :: sampling_matrix
double precision :: ran_real, omega_max, omega_min
double precision, dimension(:,:), allocatable :: omega_norm

!Set seed for the random number generator between [0,1]
CALL RANDOM_SEED(seed)


!Normalize omega
allocate(omega_norm(0:nx-1,0:ny-1))
omega_max = maxval(omega)
omega_min = minval(omega)

do j = 0,ny-1
  do i = 0,nx-1
    omega_norm(i,j) = (omega(i,j)-omega_min)/(omega_max-omega_min)
  end do
end do


!$OMP PARALLEL DO
do sample = 0,n_samples-1

  !Real random number - we avoid boundaries
  call RANDOM_NUMBER(ran_real)
  !Map to integers
  i = 1+FLOOR((nx-1)*ran_real)
  !Real random number
  call RANDOM_NUMBER(ran_real)
  !Map to integers
  j = 1+FLOOR((ny-1)*ran_real)


  sampling_matrix(sample,0) = omega_norm(i,j)
  sampling_matrix(sample,1) = omega_norm(i,j+1)
  sampling_matrix(sample,2) = omega_norm(i,j-1)
  sampling_matrix(sample,3) = omega_norm(i+1,j)
  sampling_matrix(sample,4) = omega_norm(i+1,j+1)
  sampling_matrix(sample,5) = omega_norm(i+1,j-1)
  sampling_matrix(sample,6) = omega_norm(i-1,j)
  sampling_matrix(sample,7) = omega_norm(i-1,j+1)
  sampling_matrix(sample,8) = omega_norm(i-1,j-1)

  ! sampling_matrix(sample,0) = omega(i,j)
  ! sampling_matrix(sample,1) = omega(i+1,j)
  ! sampling_matrix(sample,2) = omega(i-1,j)
  ! sampling_matrix(sample,3) = omega(i,j+1)
  ! sampling_matrix(sample,4) = omega(i+1,j+1)
  ! sampling_matrix(sample,5) = omega(i-1,j+1)
  ! sampling_matrix(sample,6) = omega(i,j-1)
  ! sampling_matrix(sample,7) = omega(i+1,j-1)
  ! sampling_matrix(sample,8) = omega(i-1,j-1)

end do
!$OMP END PARALLEL DO


deallocate(omega_norm)

return
end