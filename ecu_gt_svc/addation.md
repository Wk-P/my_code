# additional state

1. conflict condition
Random make 100 group privacy set. When place the service on the ECU, check the placed serivce and the service will be placed whether privacy conflect.

2. Change the placing constraints
    The safe constraints should change to:
    - One ECU can place many serivces (release the number of serivce limition)

    # Two violation and safe constraints
    - All vms sum of serivces placed on ECU over the max ECU capacity vms number should decide to be a violation. But P3 problem without this one. p3 the reward should be added if just the varaible `ar` more than current.
    - In placing on the ECU scheduling placement, if the services which will be placed on the ECU and have been  placed on the ECU, these services are in privacy conflect set, then this condition is a violation.


3. Conflict sets
Conflict set is a set that has K number subset, every subset have J number services.
Services in one subset will be noticed that they are violation, and can't place the service_j_{a} on the ECU which has had the service_j_{b}
the length of subset is greater and equal 2, fewer and equal sum of services.
{
    {svc1, svc2},
    {svc2, svc3, svc8}
    ...
}


