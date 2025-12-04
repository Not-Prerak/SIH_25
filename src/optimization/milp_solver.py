from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class MILPSolver:
    def __init__(self, horizon_hours: int = 6, time_step: int = 5):
        self.horizon_hours = horizon_hours
        self.time_step = time_step  # minutes
        self.problem = None
        self.time_slots = None
        
    def create_time_slots(self, start_time: datetime):
        """Convert continuous time to discrete time slots"""
        slots = {}
        current = start_time
        slot_index = 0
        
        for _ in range(int(self.horizon_hours * 60 / self.time_step)):
            slots[slot_index] = current
            current += timedelta(minutes=self.time_step)
            slot_index += 1
            
        return slots
    
    def solve(self, trains: List, stations: List, track_segments: List, 
              current_time: datetime) -> Dict:
        """
        Solve the train scheduling problem using MILP
        
        Returns: Dictionary with optimized schedule
        """
        logger.info(f"Starting MILP optimization for {len(trains)} trains")
        
        # Create time slots
        self.time_slots = self.create_time_slots(current_time)
        num_slots = len(self.time_slots)
        
        # Initialize problem
        self.problem = LpProblem("Train_Scheduling", LpMinimize)
        
        # Decision variables
        arrival_vars = {}
        departure_vars = {}
        segment_occupancy = {}
        precedence_vars = {}
        
        # Create variables for each train and station
        for train in trains:
            for station_id in train.route_station_ids:
                # Arrival time at station (continuous)
                arrival_vars[(train.id, station_id)] = LpVariable(
                    f"arr_{train.id}_{station_id}", 
                    lowBound=0, 
                    upBound=num_slots
                )
                
                # Departure time from station
                departure_vars[(train.id, station_id)] = LpVariable(
                    f"dep_{train.id}_{station_id}", 
                    lowBound=0, 
                    upBound=num_slots
                )
        
        # Create precedence variables for each pair of trains at each segment
        for i, train1 in enumerate(trains):
            for j, train2 in enumerate(trains):
                if i < j:
                    for segment in track_segments:
                        if (segment.from_station in train1.route_station_ids and 
                            segment.to_station in train1.route_station_ids and
                            segment.from_station in train2.route_station_ids and 
                            segment.to_station in train2.route_station_ids):
                            
                            # Binary variable: 1 if train1 precedes train2 on segment
                            precedence_vars[(train1.id, train2.id, segment.id)] = LpVariable(
                                f"prec_{train1.id}_{train2.id}_{segment.id}",
                                cat='Binary'
                            )
        
        # Objective function: Minimize weighted delay
        objective = lpSum([
            train.priority * (arrival_vars[(train.id, train.destination_station_id)] - 
                             self._time_to_slot(train.scheduled_arrival, current_time))
            for train in trains
        ])
        
        self.problem += objective
        
        # Constraints
        self._add_travel_time_constraints(trains, track_segments, arrival_vars, departure_vars)
        self._add_headway_constraints(trains, track_segments, precedence_vars, 
                                     arrival_vars, departure_vars)
        self._add_station_capacity_constraints(trains, stations, arrival_vars, departure_vars)
        self._add_precedence_constraints(trains, track_segments, precedence_vars, 
                                        arrival_vars, departure_vars)
        
        # Solve
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=30)
        self.problem.solve(solver)
        
        logger.info(f"Optimization status: {LpStatus[self.problem.status]}")
        
        # Extract solution
        schedule = self._extract_solution(trains, arrival_vars, departure_vars, 
                                         self.time_slots)
        
        return schedule
    
    def _add_travel_time_constraints(self, trains, track_segments, arrival_vars, departure_vars):
        """Add travel time constraints between stations"""
        for train in trains:
            route = train.route_station_ids
            for i in range(len(route) - 1):
                from_station = route[i]
                to_station = route[i+1]
                
                # Find segment between stations
                segment = next((s for s in track_segments 
                               if s.from_station == from_station and s.to_station == to_station), None)
                
                if segment:
                    min_travel = segment.min_travel_time / self.time_step
                    max_travel = segment.max_travel_time / self.time_step
                    
                    # Arrival at to_station >= departure from from_station + min travel time
                    self.problem += (
                        arrival_vars[(train.id, to_station)] >= 
                        departure_vars[(train.id, from_station)] + min_travel
                    )
                    
                    # Maximum travel time constraint
                    self.problem += (
                        arrival_vars[(train.id, to_station)] <= 
                        departure_vars[(train.id, from_station)] + max_travel
                    )
    
    def _add_headway_constraints(self, trains, track_segments, precedence_vars, 
                                arrival_vars, departure_vars):
        """Add safety headway constraints"""
        min_headway = 3  # 3 time slots = 15 minutes
        
        for segment in track_segments:
            trains_on_segment = [
                t for t in trains 
                if segment.from_station in t.route_station_ids 
                and segment.to_station in t.route_station_ids
            ]
            
            for i, train1 in enumerate(trains_on_segment):
                for train2 in trains_on_segment:
                    if train1.id != train2.id:
                        # Large M constraint for precedence
                        M = 1000  # Large number
                        prec_var = precedence_vars.get((train1.id, train2.id, segment.id))
                        
                        if prec_var:
                            # If train1 precedes train2, train2 departure >= train1 arrival + headway
                            self.problem += (
                                departure_vars[(train2.id, segment.from_station)] >=
                                arrival_vars[(train1.id, segment.to_station)] + min_headway - 
                                M * (1 - prec_var)
                            )
                            
                            # If train2 precedes train1, train1 departure >= train2 arrival + headway
                            self.problem += (
                                departure_vars[(train1.id, segment.from_station)] >=
                                arrival_vars[(train2.id, segment.to_station)] + min_headway - 
                                M * prec_var
                            )
    
    def _time_to_slot(self, dt: datetime, start_time: datetime) -> float:
        """Convert datetime to time slot index"""
        delta = dt - start_time
        minutes = delta.total_seconds() / 60
        return minutes / self.time_step
    
    def _extract_solution(self, trains, arrival_vars, departure_vars, time_slots):
        """Extract schedule from solution"""
        schedule = {}
        
        for train in trains:
            train_schedule = {}
            for station_id in train.route_station_ids:
                arr_var = arrival_vars.get((train.id, station_id))
                dep_var = departure_vars.get((train.id, station_id))
                
                if arr_var and dep_var:
                    arr_slot = arr_var.varValue
                    dep_slot = dep_var.varValue
                    
                    if arr_slot is not None and dep_slot is not None:
                        # Convert slots back to datetime
                        arr_time = time_slots.get(int(arr_slot))
                        dep_time = time_slots.get(int(dep_slot))
                        
                        if arr_time and dep_time:
                            train_schedule[station_id] = {
                                'arrival': arr_time,
                                'departure': dep_time,
                                'dwell_minutes': (dep_time - arr_time).total_seconds() / 60
                            }
            
            schedule[train.id] = train_schedule
        
        return schedule
