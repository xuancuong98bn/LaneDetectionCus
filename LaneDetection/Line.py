import numpy as np

class Line():
    def __init__(self):

        # polynomial coefficients averaged over the last n iterations
        self.best_fit_px = None
        #polynomial coefficients for the most recent fit
        self.current_fit_px = None
        # Previous Fits
        self.previous_fits_px = []
        # Limit x dimention
        self.curent_limitx = None
        self.limitx = [0, 0]

        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        
    def run_line_pipe(self):
        self.calc_best_fit()
        self.limitx[0] = min(self.curent_limitx) if self.limitx[0] == 0 else np.mean((self.limitx[0], min(self.curent_limitx)))
        self.limitx[1] = max(self.curent_limitx) if self.limitx[1] == 0 else np.mean((self.limitx[1], max(self.curent_limitx)))

    def __get_line__(self):
        return self.best_fit_px

    def __get_limitx__(self):
        return (int(self.limitx[0]), int(self.limitx[1]))

    def __add_new_fit__(self, new_fit_px, limitx):
        if self.current_fit_px is None and self.previous_fits_px == []:
            self.current_fit_px = new_fit_px
            self.curent_limitx = limitx
            self.run_line_pipe()
            return
        else:
            # measure the diff to the old fit
            self.diffs = np.abs(new_fit_px - self.current_fit_px)
            # check the size of the diff
            if self.diff_check():
                print("Found a fit diff that was too big")
                return
            self.current_fit_px = new_fit_px
            self.curent_limitx = limitx
            self.run_line_pipe()
            return

            
    def diff_check(self):
        if self.diffs[0] > 0.01:
            return False
        if self.diffs[1] > 2.5:
            return False
        if self.diffs[2] > 1000.:
            return False
        return True

    def calc_best_fit(self):
        """
        calculate the average, if needed
        """
        # add the latest fit to the previous fit list
        self.previous_fits_px.append(self.current_fit_px)

        # If we currently have 5 fits, throw the oldest out
        if len(self.previous_fits_px) > 5:
            self.previous_fits_px = self.previous_fits_px[1:]

        # Just average everything
        self.best_fit_px = np.average(self.previous_fits_px, axis=0)
        return


    # def calc_radius(self):
    #     """
    #     left_fit and right_fit are assumed to have already been converted to meters
    #     """
    #     y_eval = self.y_eval
    #     fit = self.best_fit_m
    #     if y_eval and fit:
    #         curve_rad = ((1 + (2*fit[0]*y_eval + fit[1])**2)**1.5) / np.absolute(2*fit[0])
    #         self.radius_of_curvature = curve_rad
    #     return