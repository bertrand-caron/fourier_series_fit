class Fit_Error(Exception):
    pass

class Discountinuity_Error(Fit_Error):
    pass

class Not_Enough_Points(Fit_Error):
    pass

class Duplicated_Values(Fit_Error):
    pass
