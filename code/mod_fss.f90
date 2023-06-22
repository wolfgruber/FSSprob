! Created on  10.03.2022
! Last update 22.06.2023
!
! author: Ludwig Wolfgruber
! e-mail: ludwig.wolfgruber@gmx.at

MODULE mod_fss

IMPLICIT NONE

CONTAINS

SUBROUTINE integral_table (n1, n2, array, table)
    ! computes the integral lookup table, as the cumulative sum along two
    ! dimensions. Behaviour like array.cumsum(axis=0).cumsum(axis=1) in numpy
    ! python
    INTEGER, INTENT(IN) :: n1, n2
    INTEGER             :: i, j
    REAL, INTENT(IN)    :: array(n1, n2)
    REAL, INTENT(OUT)   :: table(n1, n2)

    table(1,:) = array(1,:)

    DO i=2,n1
        table(i,:) = array(i,:) + table(i-1,:)
    END DO

    DO j=2,n2
        table(:,j) = table(:,j) + table(:, j-1)
    END DO

END SUBROUTINE integral_table


SUBROUTINE integral_filter (n1, n2, kernel, field_in, field_out)
    ! look up process in table (field_in). field_in has to be processed by 
    ! integral_table. Behaviour like scipy.signal.convolve2d(array, kernel,
    ! mode='same') in python scipy
    INTEGER, INTENT(IN) :: n1, n2, kernel
    INTEGER             :: r0(n1), r1(n1), c0(n2), c1(n2), radius, i, j
    REAL, INTENT(IN)    :: field_in(n1, n2)
    REAL, INTENT(OUT)   :: field_out(n1, n2)
    
    radius = FLOOR(kernel/2.)

    IF (MOD(kernel, 2) == 0) THEN ! case even sized kernel
        DO i=1, n1
            r0(i) = MAX(1, i-radius)
            r1(i) = MIN(n1, i+radius)
        END DO
        
        DO j=1, n2
            c0(j) = MAX(1, j-radius)
            c1(j) = MIN(n2, j+radius)
        END DO
    
    ELSE ! case odd sized kernel
        DO i=1, n1
            r0(i) = MAX(1, i-radius)
            r1(i) = MIN(n1, i+radius+1)
        END DO
        
        DO j=1, n2
            c0(j) = MAX(1, j-radius)
            c1(j) = MIN(n2, j+radius+1)
        END DO
    
    END IF
    field_out = field_in(r0, c0) + field_in(r1, c1) &
              - field_in(r0, c1) - field_in(r1, c0)
END SUBROUTINE integral_filter


SUBROUTINE compute_fss_table (n1, n2, kernel, field1, field2, fss)
    ! compute fss for precomputed tables. field1 and field2 have to be
    ! preprocessed by integral_table
    INTEGER, INTENT(IN) :: n1, n2, kernel
    REAL, INTENT(IN)    :: field1(n1, n2), field2(n1, n2)
    REAL, INTENT(OUT)   :: fss
    REAL                :: numinator, denominator
    REAL                :: fhat1(n1, n2), fhat2(n1, n2)
    
    CALL integral_filter(n1, n2, kernel, field1, fhat1)
    CALL integral_filter(n1, n2, kernel, field2, fhat2)
    
    numinator = SUM( (fhat1 - fhat2)**2 )
    denominator = SUM( fhat1**2 + fhat2**2 )
    
    fss = 1. - numinator / denominator
END SUBROUTINE compute_fss_table


SUBROUTINE compute_fss (n1, n2, kernel, field1, field2, fss)
    ! compute the FSS for one kernel size. The variables field1 and field2 have
    ! to contain probability values, either binary for deterministic case or in
    ! [0,1] for probabilisitc case. If different kernel sizes are computed
    ! fss_one_thrsh is faster, as it reuses the integral tables.
    INTEGER, INTENT(IN) :: n1, n2, kernel
    REAL, INTENT(IN)    :: field1(n1, n2), field2(n1, n2)
    REAL, INTENT(OUT)   :: fss
    REAL                :: numinator, denominator
    REAL                :: table1(n1, n2), table2(n1, n2)
    REAL                :: fhat1(n1, n2), fhat2(n1, n2)
    
    CALL integral_table (n1, n2, field1, table1)
    CALL integral_table (n1, n2, field2, table2)
      
    CALL integral_filter(n1, n2, kernel, table1, fhat1)
    CALL integral_filter(n1, n2, kernel, table2, fhat2)
    
    numinator = SUM( (fhat1 - fhat2)**2 )
    denominator = SUM( fhat1**2 + fhat2**2 )
    
    fss = 1. - numinator / denominator
END SUBROUTINE compute_fss


SUBROUTINE fss_one_thrsh (n1, n2, nkernel, kernel, field1, field2, fss)
    ! compute the FSS for nkernel kernel sizes. The variables field1 and field2
    ! have to contain probability values, either binary for deterministic case 
    ! or in [0,1] for probabilisitc case.
    INTEGER, INTENT(IN) :: n1, n2, nkernel
    INTEGER, INTENT(IN) :: kernel(nkernel)
    INTEGER :: i
    REAL, INTENT(IN)    :: field1(n1, n2), field2(n1, n2)
    REAL, INTENT(OUT)   :: fss(nkernel)
    REAL                :: numinator, denominator
    REAL                :: table1(n1, n2), table2(n1, n2)
    
    CALL integral_table (n1, n2, field1, table1)
    CALL integral_table (n1, n2, field2, table2)
    
    DO i = 1,nkernel
        IF (kernel(i) <= 1) THEN
            numinator = SUM( (field1 - field2)**2 )
            denominator = SUM( field1**2 + field2**2 )

            fss(i) = 1. - numinator / denominator
        ELSE
            CALL compute_fss_table(n1, n2, kernel(i), table1, table2, fss(i))
        END IF
    END DO 
END SUBROUTINE fss_one_thrsh


SUBROUTINE fss_prob (n1, n2, n_ens, n_thrsh, n_kernel, fcst, obs, thrsh, kernel, fss)
    ! compute the FSSprob for one (ensemble forecast, observation) pair and
    ! different thresholds and kernel sizes. The field forecast has dimensions
    ! (ensemble size, x, y) and contains real forcast variables.
    INTEGER, INTENT(IN) :: n1, n2, n_ens, n_thrsh, n_kernel
    INTEGER, INTENT(IN) :: kernel(n_kernel)
    INTEGER             :: i, j
    REAL, INTENT(IN)    :: fcst(n_ens, n1, n2), obs(n1, n2), thrsh(n_thrsh)
    REAL                :: fcst_field(n1, n2), obs_field(n1, n2)
    REAL                :: ens_field(n1, n2)
    REAL, INTENT(OUT)   :: fss(n_thrsh, n_kernel)
    
    fss  = 0.

    DO i=1, n_thrsh
        WHERE (obs >= thrsh(i))
            obs_field = 1.
        ELSEWHERE
            obs_field = 0.
        END WHERE
                
        fcst_field = 0.
        DO j=1, n_ens
            WHERE (fcst(j,:,:) >= thrsh(i))
                ens_field = 1.
            ELSEWHERE
                ens_field = 0.
            END WHERE
            fcst_field = fcst_field + ens_field
        END DO
        fcst_field = fcst_field / n_ens
                
        CALL fss_one_thrsh (n1, n2, n_kernel, kernel, fcst_field, obs_field, fss(i,:))
    END DO
   
END SUBROUTINE fss_prob


SUBROUTINE ensemble_fss_one_lead_time (n1, n2, n_ens, n_ens_size, n_thrsh, n_kernel, fcst, obs, ens_size, thrsh, kernel, fss)
    ! compute the FSSprob for one (ensemble forecast, observation) pair and
    ! different ensemble subsamples, thresholds and kernel sizes. A setup like
    ! this was used to produce results for the study "The fractions skill score
    ! for ensemble forecast verification". Do not use for regular verification!
    INTEGER, INTENT(IN) :: n1, n2, n_ens, n_ens_size, n_thrsh, n_kernel
    INTEGER, INTENT(IN) :: ens_size(n_ens_size), kernel(n_kernel)
    INTEGER             :: i, j, k, l, sample(n_ens_size), ens_idx
    REAL, INTENT(IN)    :: fcst(n_ens, n1, n2), obs(n1, n2), thrsh(n_thrsh)
    REAL                :: fcst_field(n1, n2), obs_field(n1, n2)
    REAL                :: ens_field(n1,n2)
    REAL, ALLOCATABLE   :: mean_fss(:,:,:)
    REAL, INTENT(OUT)   :: fss(n_ens_size, n_thrsh, n_kernel)
    
    fss  = 0.
    
    sample = 960 / ens_size
    
    DO i=1, n_ens_size
        ALLOCATE(mean_fss(sample(i), n_thrsh, n_kernel))
        ens_idx = 1
        
        DO j=1, sample(i)
            DO k=1, n_thrsh
                WHERE (obs >= thrsh(k))
                    obs_field = 1.
                ELSEWHERE
                    obs_field = 0.
                END WHERE
                
                fcst_field = 0.
                DO l=ens_idx, ens_idx + ens_size(i) - 1
                    WHERE (fcst(l,:,:) >= thrsh(k))
                        ens_field = 1.
                    ELSEWHERE
                        ens_field = 0.
                    END WHERE
                    
                    fcst_field = fcst_field + ens_field
                END DO
                fcst_field = fcst_field / ens_size(i)
                
                CALL fss_one_thrsh (n1, n2, n_kernel, kernel, fcst_field, obs_field, mean_fss(j,k,:))
            END DO
            ens_idx = ens_idx + ens_size(i)
        END DO
        
        DO l=1, sample(i)
            fss(i,:,:) = fss(i,:,:) + mean_fss(l,:,:)
        END DO
        fss(i,:,:) = fss(i,:,:) / sample(i)
        
        DEALLOCATE(mean_fss)
        
    END DO
    
END SUBROUTINE ensemble_fss_one_lead_time

END MODULE mod_fss
