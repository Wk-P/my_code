class ECU:
    def __init__(self, name, capacity):
        # capacity is the number of VMs that the ECU has
        self.name = name
        self.capacity = capacity
    
    def __str__(self):
        return f"ECU(name={self.name}, capacity={self.capacity})"
    

    def __dict__(self):
        return {"name": self.name, "capacity": self.capacity}
    
class SVC:
    def __init__(self, name, requirement):
        # requirement is the number of VMs that the SVC needs
        self.name = name
        self.requirement = requirement
    
    def __str__(self):
        return f"SVC(name={self.name}, requirement={self.requirement})"
    
    def __dict__(self):
        return {"name": self.name, "requirement": self.requirement}