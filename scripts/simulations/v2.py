import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from dataclasses import dataclass, asdict


def deg_to_rad(theta):
    return theta * np.pi / 180

def rad_to_deg(theta):
    return theta * 180 / np.pi

def cot(theta):
    return 1 / np.tan(theta)

def tan(theta):
    return np.tan(theta)

def atan(theta):
    return np.atan(theta)

def sin(theta):
    return np.sin(theta)

def cos(theta):
    return np.cos(theta)


@dataclass
class SimulationConfig:
    L_s: float = 10.2  # speaker membrane diameter (cm)
    W_s: float = 4.5  # speaker width (cm)
    C_s: float = 5.0   # speaker redirection cone length (cm)
    L_t: float = 20.0  # target diameter (cm)

    H_l: float = 0.0 # height from base to laser pointer center (cm)
    H_s: float = 6.8   # height from base to membrane center (cm)
    H_s_top: float = 12.6  # height from base to speaker top (cm)
    h_tolerance: float = 7.4  # upper height tolerance
    # heta_min: float = 6/L_t
    # heta_max: float = 18/L_t
    heta_min: float = 1
    heta_max: float = (L_t - 1) / 2

    # print(heta_min, heta_max)

    # step sizes
    # delta_step_deg: float = 0.05
    # D_o_t_step: float = 0.5
    # D_l_s_step: float = 0.5
    # H_m_step: float = 0.5

    delta_step_deg: float = 0.1
    D_o_t_step: float = 0.5
    D_l_s_step: float = 0.5
    H_m_step: float = 0.5


    def __post_init__(self):
        self.H_max = self.H_s_top + self.h_tolerance
        self.delta_range = np.array([0.5, 3.0])
        # self.delta_range = np.array([-3.0, -0.5])
        self.D_o_t_range = np.array([0.1, 60])
        self.D_l_s_range = np.array([0.1, 20])
        self.H_m_range = np.array([self.H_s, self.H_s_top])

        self.delta_values = np.arange(
            self.delta_range[0],
            self.delta_range[1] + self.delta_step_deg,
            self.delta_step_deg
        )
        self.D_l_s_values = np.arange(
            self.D_l_s_range[0],
            self.D_l_s_range[1] + self.D_l_s_step,
            self.D_l_s_step
        )
        self.D_o_t_values = np.arange(
            self.D_o_t_range[0],
            self.D_o_t_range[1] + self.D_o_t_step,
            self.D_o_t_step
        )
        self.H_m_values = np.arange(
            self.H_m_range[0],
            self.H_m_range[1] + self.H_m_step,
            self.H_m_step
        )


class Simulator:
    def __init__(self, config: SimulationConfig):
        self.cfg = config

    def get_values(self, delta, d_l_s, d_o_t, h_m):
        cfg = self.cfg

        # calculate theta value based on laser distance and base heights
        theta = atan((cfg.H_s - cfg.H_l) / d_l_s)

        # to reflect perpendicularly
        # phi = np.pi/4 + theta/2

        phi = np.pi/4 + theta/2

        y = h_m - cfg.H_s

        # mirror speaker distance
        d_m_s = y * cot(theta)

        # first mirror perpendicular height
        # mh1 = y * (1 - cot(theta) * tan(theta + delta)) / (1 - tan(phi) * tan(theta + delta))

        mh1 = y * cot(theta) * tan(theta + delta) / (1 + tan(phi) * tan(theta + delta))


        # first mirror horizontal delta distance
        dd1 = mh1 * tan(phi)

        # second mirror horizontal delta distance
        # dd2 = (dd1 - tan(delta) * (h_m - cfg.H_s + mh1)) / (1 - tan(delta))

        dd2 = (dd1 - (y + mh1) * tan(delta)) / (1 - tan(delta))

        # second mirror perpendicular height (equal since 45 degrees)
        mh2 = dd2

        # minimum mirror lengths
        ml_min_1 = 2 * (dd1 / sin(phi))
        ml_min_2 = 2 * dd2 * np.sqrt(2)

        # for positive deviation add dd2
        d_m_t = d_o_t + dd2

        # print(mh, dd, ml_min)
        # dd = (h_m - cfg.H_s) * (cot(theta) - cot(theta + delta))
        # mh = dd * cot(phi)
        # ml_min = 2 * (dd / sin(phi))
        #


        # max mirror to opening distance
        d_m_o_max = (cfg.H_s_top - cfg.H_s + mh2) / tan(delta)

        # if cfg.H_max - h_m + mh < 0:
        #     print(cfg.H_max, h_m, mh)

        # calculate coverage
        cov = d_m_t * tan(delta) + mh2

        return {
            "theta": rad_to_deg(theta),
            "phi": rad_to_deg(phi),
            "delta": rad_to_deg(delta),
            "laser_speaker_distance": d_l_s,
            "mirror_speaker_distance": d_m_s,
            "mirror_target_distance": d_m_t,
            "mirror_height": h_m,
            "min_mirror_length_1": ml_min_1,
            "min_mirror_length_2": ml_min_2,
            "max_opening_distance": d_m_o_max,
            "coverage": cov,
        }


    def satisfies_constraints(self, vals):
        cfg = self.cfg
        return (
            # coverage within bounds
            cfg.heta_min <= vals["coverage"] <= cfg.heta_max and
            # mirror vertical doesnt exceed max height
            (vals["mirror_height"] + (vals["min_mirror_length_1"] * tan(deg_to_rad(vals["theta"])/2) / 2)) <= cfg.H_max
        )


    def calculate_configs(self):
        cfg = self.cfg

        param_grid = product(
            cfg.delta_values,
            cfg.D_l_s_values,
            cfg.D_o_t_values,
            cfg.H_m_values
        )

        total = (
                len(cfg.delta_values)
                * len(cfg.D_l_s_values)
                * len(cfg.D_o_t_values)
                * len(cfg.H_m_values)
        )

        results = []

        for delta, d_l_s, d_o_t, h_m in tqdm(param_grid, total=total, desc="Simulating", ncols=90):
            delta_rad = np.deg2rad(delta)
            vals = self.get_values(delta_rad, d_l_s, d_o_t, h_m)

            if self.satisfies_constraints(vals):
                results.append(vals)

        return pd.DataFrame(results).round(2)

    def group_configs(self, df, delta_range):

        print("Grouping...")

        df = df.groupby([
            'theta', 'phi', 'laser_speaker_distance', 'mirror_speaker_distance', 'mirror_height'
        ])
        df = df.agg({
            'delta': ['min', 'max'],
            'coverage': ['min', 'max'],
            'min_mirror_length_1': 'max',
            'min_mirror_length_2': 'max',
            'max_opening_distance': 'min'
        })

        df.columns = ['_'.join(col) for col in df.columns]
        df = df.reset_index()

        # filter
        df = df[
            (df['delta_min'] == delta_range[0]) & (df['delta_max'] == delta_range[1])
            ]

        df['delta_diff'] = df['delta_max'] - df['delta_min']
        df['coverage_diff'] = df['coverage_max'] - df['coverage_min']
        # df['score'] = 0.7 * df['coverage_diff'] + 0.3 * df['min_mirror_length_max']

        # drop degenerate ranges (no variation)
        df = df[df['delta_diff'] > 0]
        df = df[df['coverage_diff'] > 0]

        # sort by score (higher = better)
        df = df.sort_values(['coverage_diff'], ascending=False).reset_index(drop=True).round(2)

        print("Done.")

        return df


# === Run Simulation === #
if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    cfg = SimulationConfig()
    sim = Simulator(cfg)

    observed_delta_range = np.array([0.5, 3.0])


    do_simulate = False

    if do_simulate:
        configs_df = sim.calculate_configs()
        print("Saving simulation results...")
        configs_df.to_csv("../data/simulation_results.csv", index=False)
        print("Done.")
    else:
        print("Loading simulation results from disk...")
        configs_df = pd.read_csv("../../data/simulation_results.csv")
        print("Done.")

    configs_df = configs_df.sort_values(['theta'], ascending=True)
    # configs_df = configs_df.sort_values(['coverage'], ascending=False)

    print("\n--- Simulation Results ---")
    print(configs_df.head(100))
    print(f"\nTotal combinations computed: {len(configs_df)}")

    grouped_df = sim.group_configs(configs_df, observed_delta_range)
    grouped_df.to_csv("../data/grouped_simulation_results.csv", index=False)

