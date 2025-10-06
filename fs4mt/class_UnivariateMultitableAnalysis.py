from tqdm import tqdm
from khiops import core as kh
import re
import os
import pandas as pd
import sys
import json
from sys import exit

# Add the path to the parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "")))

# Get Khiops version (to manage the differences between v10 and v11)
large_version = kh.get_khiops_version()
version = str(large_version).split(".")[0]
if version == "10":
    kh_version = "v10"
elif version == "11":
    kh_version = "v11"
else:
    print("Error : Version " + version + " of Khiops is not supported by this tool.")
    exit()

# Dictionary to match TablePartiton with construction rule
Dict_matching_partition = {
    "TablePartitionCount": "TableCount",
    "TablePartitionCountDistinct": "TableCountDistinct",
    "TablePartitionMode": "TableMode",
    "TablePartitionModeAt": "TableModeAt",
    "TablePartitionMean": "TableMean",
    "TablePartitionStdDev": "TableStdDev",
    "TablePartitionMedian": "TableMedian",
    "TablePartitionMin": "TableMin",
    "TablePartitionMax": "TableMax",
    "TablePartitionSum": "TableSum",
}

# Dictionary to match primitive name with construction rule
Dict_matching_rule_name = {
    "Count": "TableCount",
    "CountDistinct": "TableCountDistinct",
    "Mode": "TableMode",
    "ModeAt": "TableModeAt",
    "Mean": "TableMean",
    "StdDev": "TableStdDev",
    "Median": "TableMedian",
    "Min": "TableMin",
    "Max": "TableMax",
    "Sum": "TableSum",
    "Date": "GetDate",
    "Time": "GetTime",
}


def get_key(dictionary, val):
    """Get the key of a value in a python dictionary

    :param dictionary: Python dictionary.
    :type dictionary: dict
    :param val: Value to test.
    :return: The key associated to the value

    """
    for key, value in dictionary.items():
        if value == val:
            return key
    return -1


def add_noise(dictionary_domain, number_noise):
    """Add noise variable into secondaries tables.

    :param dictionary_domain: Khiops dictionary domain
    :param number_noise: Number of noise variable to add per table
    :type number_noise: int
    :return: Khiops dictionary domain with noise
    """
    # Initialize
    index = 0
    noise_variable = []
    for dictionary in dictionary_domain.dictionaries:
        if not dictionary.root:
            number_add_noise = 0
            for i in range(number_noise):
                # Add numerical noise variable
                if number_add_noise < number_noise:
                    variable = kh.Variable()
                    variable.name = "N_" + str(index)
                    noise_variable.append(variable.name)
                    variable.type = "Numerical"
                    variable.used = True
                    variable.rule = "Sum(Random()," + str(i) + ")"
                    dictionary_domain.get_dictionary(dictionary.name).add_variable(
                        variable
                    )
                    noise_variable.append(variable.name)
                    number_add_noise += 1
                # Add categorical noise variable
                if number_add_noise < number_noise:
                    variable = kh.Variable()
                    variable.name = "C_" + str(index)
                    noise_variable.append(variable.name)
                    variable.type = "Categorical"
                    variable.used = True
                    variable.rule = (
                        'Concat("V_",FormatNumerical(Round(Product('
                        + str(2 ** (i + 1))
                        + ",Random())),0,0))"
                    )
                    dictionary_domain.get_dictionary(dictionary.name).add_variable(
                        variable
                    )
                    noise_variable.append(variable.name)
                    number_add_noise += 1
                index += 1
                if number_add_noise >= number_noise:
                    break
    return dictionary_domain, noise_variable


def get_dataset(dataset_name):
    if dataset_name == "Accident_star":
        # Set accident data information
        data_path = os.path.join("DATA", "Accident")
        dictionary_file_path = os.path.join(data_path, "Accidents_etoile.kdic")
        data_table_path = os.path.join(data_path, "Accidents.txt")
        vehicle_table_path = os.path.join(data_path, "Vehicles.txt")
        user_table_path = os.path.join(data_path, "Users.txt")
        place_table_path = os.path.join(data_path, "Places.txt")
        main_dictionary_name = "Accident"
        Additional_data_tables = {
            main_dictionary_name + "`Place": place_table_path,
            main_dictionary_name + "`Vehicles": vehicle_table_path,
            main_dictionary_name + "`Users": user_table_path,
        }
        target = "Gravity"
        return (
            data_path,
            dictionary_file_path,
            data_table_path,
            [vehicle_table_path, user_table_path, place_table_path],
            Additional_data_tables,
            main_dictionary_name,
            target,
        )

    if dataset_name == "Accident_star_TableSelection":
        # Set accident data information
        data_path = os.path.join("DATA", "Accident")
        dictionary_file_path = os.path.join(
            data_path, "Accidents_etoile_TableSelection.kdic"
        )
        data_table_path = os.path.join(data_path, "Accidents.txt")
        vehicle_table_path = os.path.join(data_path, "Vehicles.txt")
        user_table_path = os.path.join(data_path, "Users.txt")
        place_table_path = os.path.join(data_path, "Places.txt")
        main_dictionary_name = "Accident"
        Additional_data_tables = {
            main_dictionary_name + "`Place": place_table_path,
            main_dictionary_name + "`Vehicles": vehicle_table_path,
            main_dictionary_name + "`Users": user_table_path,
        }
        target = "Gravity"
        return (
            data_path,
            dictionary_file_path,
            data_table_path,
            [vehicle_table_path, user_table_path, place_table_path],
            Additional_data_tables,
            main_dictionary_name,
            target,
        )

    if dataset_name == "Accident_flocon":
        # Set accident data information
        data_path = os.path.join("DATA", "Accidents")
        dictionary_file_path = os.path.join(data_path, "Accidents.kdic")
        data_table_path = os.path.join(data_path, "Accidents.txt")
        vehicle_table_path = os.path.join(data_path, "Vehicles.txt")
        user_table_path = os.path.join(data_path, "Users.txt")
        place_table_path = os.path.join(data_path, "Places.txt")
        main_dictionary_name = "Accident"
        Additional_data_tables = {
            main_dictionary_name + "`Place": place_table_path,
            main_dictionary_name + "`Vehicles": vehicle_table_path,
            main_dictionary_name + "`Vehicles`Users": user_table_path,
        }
        target = "Gravity"
        return (
            data_path,
            dictionary_file_path,
            data_table_path,
            [vehicle_table_path, user_table_path, place_table_path],
            Additional_data_tables,
            main_dictionary_name,
            target,
        )

    if dataset_name == "Accident_flocon_inversion_user_vehicle":
        # Set accident data information
        data_path = os.path.join("DATA", "Accident")
        dictionary_file_path = os.path.join(data_path, "Accidents_flocon.kdic")
        data_table_path = os.path.join(data_path, "Accidents.txt")
        vehicle_table_path = os.path.join(data_path, "Vehicles.txt")
        user_table_path = os.path.join(data_path, "Users.txt")
        place_table_path = os.path.join(data_path, "Places.txt")
        main_dictionary_name = "Accident"
        Additional_data_tables = {
            main_dictionary_name + "`Place": place_table_path,
            main_dictionary_name + "`Users`Vehicles": vehicle_table_path,
            main_dictionary_name + "`Users": user_table_path,
        }
        target = "Gravity"
        return (
            data_path,
            dictionary_file_path,
            data_table_path,
            [vehicle_table_path, user_table_path, place_table_path],
            Additional_data_tables,
            main_dictionary_name,
            target,
        )

    if dataset_name == "synth1":
        # Set data information
        data_path = os.path.join("DATA", "synth1")
        dictionary_file_path = os.path.join(data_path, "sample2_synthetic_dm_ids.kdic")
        data_table_path = os.path.join(data_path, "sample2_synthetic_ids.csv")
        datamart_path = os.path.join(data_path, "sample2_synthetic_DM1_07.csv")
        logs_table_path = os.path.join(data_path, "sample2_synthetic_log.csv")

        main_dictionary_name = "TablePrincipale"
        Additional_data_tables = {
            main_dictionary_name + "`DataMart1": datamart_path,
            main_dictionary_name + "`LOGS": logs_table_path,
        }
        target = "TARGET"
        return (
            data_path,
            dictionary_file_path,
            data_table_path,
            [datamart_path, logs_table_path],
            Additional_data_tables,
            main_dictionary_name,
            target,
        )


class UnivariateMultitableAnalysis:
    """Estimates the importance of secondary native variables in multi-table
    data on the target variable using a univariate approach with or without
    discretization.

    :param dictionary_file_path: Path of a Khiops dictionary file.
    :type dictionary_file_path: str
    :param dictionary_name: Name of the dictionary to be analyzed.
    :type dictionary_name: str
    :param data_table_path: Path of the data table file.
    :type data_table_path: str
    :param additional_data_tables: A dictionary containing the data paths and file paths for a multi-table dictionary file.
    :type additional_data_tables: dict
    :param target_variable: Name of the target variable.
    :type target_variable: str
    :param exploration_type: Parameter to be analyze, 'All' for both variable and primitive, 'Variable' or 'Primitive for only variable or primitive,  defaults to 'Variable'.
    :type exploration_type: str
    :param count_effect_reduction: State of discretization, True is used, defaults to True.
    :type count_effect_reduction: bool
    :param max_trees: Maximum number of trees to construct, defaults to 0.
    :type max_trees: int, optional
    :param max_constructed_variables_per_variable: Maximum number of variables to construct per native variable, defaults to 10.
    :type max_constructed_variables_per_variable: int, optional
    :param results_dir: Path of the results directory, defaults to "Results".
    :type results_dir: str, optional
    :param construction_rules: Allowed rules for the automatic variable construction, defaults to kh.all_construction_rules.
    :type construction_rules: list, optional
    :param output_dir: Path of the output directory, defaults to "".
    :type output_dir: str, optional
    """

    def __init__(
        self,
        dictionary_file_path,
        dictionary_name,
        data_table_path,
        additional_data_tables,
        target_variable,
        exploration_type="Variable",
        count_effect_reduction=True,
        max_trees=0,
        max_constructed_variables_per_variable=10,
        construction_rules=kh.all_construction_rules,
        results_dir="Results",
        output_dir="",
    ):
        """
        Initialize class
        """

        # Dictionary attributes
        self.dictionary_path = dictionary_file_path  # Khiops dictionary path
        self.dictionary_domain = kh.read_dictionary_file(
            dictionary_file_path
        )  # Khiops dictionary domain
        self.dictionary_name = dictionary_name  # Name of the dictionary to analyze
        self.root_table_path = data_table_path  # Path of the data table
        self.additional_table = additional_data_tables  # A dictionary containing the data paths and file paths for a multi-table dictionary file
        self.target = target_variable  # Name of the target variable
        self.unused_variable = []  # List of unused variable

        # Training attributes
        self.number_aggregate = max_constructed_variables_per_variable  # Maximum number of variables to construct per native variable
        self.number_tree = max_trees  # Maximum number of trees to construct
        self.result_directory = results_dir  # Results directory path
        self.construction_rules = (
            construction_rules  # List of construction rules to use
        )

        self.output_dir = output_dir  # Output directory path
        self.discretisation = count_effect_reduction  # State of discretization option

        # Other
        if exploration_type not in ["Variable", "All", "Primitive"]:
            print(
                'Error : exploration_type must be "Variable", "All" or "Primitive" (default "Variable")'
            )
            exit()
        self.exploration_type = exploration_type  # Exploration type
        self.pattern = r"[ ,.;`()]"  # to split derivation rule and extract variables and/or primitives list
        self.match_name_dictionary_variable_name = {}  # Dictionary to match dictionary name with its variable name
        self.match_dictionary_parent_dictionary = {}  # Dictionary to match dictionary with its parent dictionary
        self.analyse_count = 0  # Real number of constructed variable
        self.table_exploration_array = (
            pd.DataFrame(  # Pandas dataframe to save tables information
                columns=[
                    "Table",
                    "Root",
                    "Variable number",
                    "Categorical variable number",
                    "Numerical variable number",
                    "Date variable number",
                ]
            )
        )

    def init_variable_importance_dictionary(self):
        """Initialize the dictionary to save the variables importance

        :return: empty dictionary
        :rtype: dict
        """
        variable_importance_dictionary = {}
        return variable_importance_dictionary

    def init_importance_list_primitive(self):
        """Initialize the importance primitive list

        :return: Initial list of importance for primitive
        :rtype: list
        """
        importance_primitives = []
        for primitive in self.construction_rules:
            importance_primitives.append([primitive, 0])
        return importance_primitives

    def init_columns_names(self):
        """Initialize the columns names"""
        self.col_type = "type"
        self.col_level = "levelMT"
        self.col_agg = "levelMT aggregate"
        self.col_nb_agg = "real number of aggregates"
        self.col_importances_list = "importances list"
        self.col_level_no_discret = "importance"
        self.col_agg_no_discret = "importance aggregate"

    def match_dictionary_name_variable_name(self):
        """
        Match each secondary dictionary name with its corresponding variable name in the parent dictionary
        and match each secondary dictionary name with its parent dictionary.
        """
        for dictionary in self.dictionary_domain.dictionaries:
            for variable in dictionary.variables:
                if (
                    variable.type == "Table" or variable.type == "Entity"
                ) and variable.used:
                    self.match_name_dictionary_variable_name[variable.name] = (
                        variable.object_type
                    )
                    self.match_dictionary_parent_dictionary[variable.object_type] = (
                        dictionary.name
                    )

    # def get_grouping_variable(self, match_datatable_table_number_in_schema):
    def get_grouping_variable(self):
        # Initialize python dictionary to match grouping intervals of count variable with tables names
        match_variable_to_add_table_name = {}

        # Train recoder on data
        if kh_version == "v10":
            train_reports_path, modeling_dictionary_path = kh.train_recoder(
                self.dictionary_path,
                self.dictionary_name,
                self.root_table_path,
                self.target,
                self.result_directory,
                additional_data_tables=self.additional_table,
                informative_variables_only=False,
                max_constructed_variables=100,
                max_trees=self.number_tree,
                construction_rules=self.construction_rules,
            )
        elif kh_version == "v11":
            train_reports_path, modeling_dictionary_path = kh.train_recoder(
                self.dictionary_path,
                self.dictionary_name,
                self.root_table_path,
                self.target,
                os.path.join(self.result_directory, "AnalysisResults.khj"),
                additional_data_tables=self.additional_table,
                informative_variables_only=False,
                max_constructed_variables=100,
                max_trees=self.number_tree,
                construction_rules=self.construction_rules,
            )

        preparation_report = kh.read_analysis_results_file(
            train_reports_path
        ).preparation_report

        # Add variable to select group

        for aggregate in preparation_report.get_variable_names():
            split_aggregate = re.split(self.pattern, aggregate)
            if split_aggregate[0] == "Count" and len(split_aggregate) == 3:
                # # Add variable for selection
                IP_count_variable = (
                    kh.read_dictionary_file(modeling_dictionary_path)
                    .get_dictionary("R_" + self.dictionary_name)
                    .get_variable("IdP" + aggregate)
                )
                P_count_variable = (
                    kh.read_dictionary_file(modeling_dictionary_path)
                    .get_dictionary("R_" + self.dictionary_name)
                    .get_variable("P" + aggregate)
                )
                count_variable = (
                    kh.read_dictionary_file(modeling_dictionary_path)
                    .get_dictionary("R_" + self.dictionary_name)
                    .get_variable(aggregate)
                )
                count = kh.Variable()
                count.name = count_variable.name
                count.type = "Numerical"
                count.used = False
                count.rule = "Sum(" + str(count_variable.rule) + ",0)"
                number_group = 1
                if (
                    preparation_report.get_variable_statistics(
                        count_variable.name
                    ).data_grid
                    is not None
                ):
                    number_group = len(
                        preparation_report.get_variable_statistics(count_variable.name)
                        .data_grid.dimensions[0]
                        .partition
                    )
                match_variable_to_add_table_name[split_aggregate[1]] = [
                    IP_count_variable,
                    P_count_variable,
                    count,
                    number_group,
                ]

        # for a direct secondary table search : Count(table_name)
        # for a second secondary table search : a_primitive(a_table_name.Count(table_name))
        # for a third secondary table search :
        #   b_primitive(b_table_name(a_primitive(a_table_name.Count(table_name))))
        """
        for table_name in match_datatable_table_number_in_schema.keys():
            for aggregate in preparation_report.get_variable_names():
                split_aggregate = re.split(self.pattern, aggregate)
                if (
                    len(split_aggregate) == (
                        3 * match_datatable_table_number_in_schema[table_name]
                        ) 
                    and ("Count("+table_name+")" in aggregate)
                ):
                    # # Add variable for selection
                    IP_count_variable = (
                        kh.read_dictionary_file(modeling_dictionary_path)
                        .get_dictionary("R_" + self.dictionary_name)
                        .get_variable("IdP" + aggregate)
                    )
                    P_count_variable = (
                        kh.read_dictionary_file(modeling_dictionary_path)
                        .get_dictionary("R_" + self.dictionary_name)
                        .get_variable("P" + aggregate)
                    )
                    count_variable = (
                        kh.read_dictionary_file(modeling_dictionary_path)
                        .get_dictionary("R_" + self.dictionary_name)
                        .get_variable(aggregate)
                    )
                    count = kh.Variable()
                    count.name = count_variable.name
                    count.type = "Numerical"
                    count.used = False
                    count.rule = "Sum(" + str(count_variable.rule) + ",0)"
                    number_group = 1
                    if (
                        preparation_report.get_variable_statistics(
                            count_variable.name
                        ).data_grid
                        is not None
                    ):
                        number_group = len(
                            preparation_report.get_variable_statistics(count_variable.name)
                            .data_grid.dimensions[0]
                            .partition
                        )
                    match_variable_to_add_table_name[
                        table_name
                    ] = [IP_count_variable, P_count_variable, count, number_group]
                    break
        """
        return match_variable_to_add_table_name

    def initialize_variable_state(self):
        """
        Initialize dictionary domain with all secondaries variables to unused
        """
        for dictionary in self.dictionary_domain.dictionaries:
            for variable in dictionary.variables:
                if not variable.used:
                    self.unused_variable.append(variable.name)
                # Set secondaries table to unused
                if dictionary.root and (
                    variable.type == "Table" or variable.type == "Entity"
                ):
                    self.dictionary_domain.get_dictionary(dictionary.name).get_variable(
                        variable.name
                    ).used = False

                elif dictionary.root and (variable.name != self.target):
                    self.dictionary_domain.get_dictionary(dictionary.name).get_variable(
                        variable.name
                    ).used = False
                # Set secondaries variable to unused
                elif not dictionary.root:
                    self.dictionary_domain.get_dictionary(dictionary.name).get_variable(
                        variable.name
                    ).used = False

    def get_primitives_in_aggregate(self, derivation_rule):
        """Get the primitives names in the aggregate's derivation rule

        :param derivation_rule: Derivation rule
        :type derivation_rule: str
        :return: List of primitive use in the derivation rule
        :rtype: list
        """
        primitive_list = []  # Init a primitive list
        # Check if a primitive is in the derivation rule
        split_derivation_rule = re.split(self.pattern, derivation_rule)
        # 1- Check if the primitive is direcly present in the derivation rule
        for primitive in self.construction_rules:
            if primitive in split_derivation_rule:
                if primitive not in primitive_list:
                    primitive_list.append(primitive)

        # 2- Check if the primitive is a TableSelection rules and match with its corresponding construction rule
        for primitive in Dict_matching_partition.keys():
            if primitive in split_derivation_rule:
                if Dict_matching_partition[primitive] not in primitive_list:
                    primitive_list.append(Dict_matching_partition[primitive])
                if "TableSelection" not in primitive_list:
                    primitive_list.append("TableSelection")

        # 3- Check if the primitive name is present in the derivation rule
        # -> primitive name may be present instead of construction rule when
        # multiples primitives are used in the derivation rule
        for primitive in Dict_matching_rule_name.keys():
            if primitive in split_derivation_rule:
                if Dict_matching_rule_name[primitive] not in primitive_list:
                    primitive_list.append(Dict_matching_rule_name[primitive])
        return primitive_list

    def update_importance_list_primitive(
        self, primitive_importance, primitive_to_update, importance
    ):
        """Update the primitive importance list according to new measured importance.

        :param primitive_to_update: List of primitive to be update
        :type primitive_to_update: list
        :param importance: Primitive importance
        :type importance: float
        """
        for i in range(len(primitive_importance)):
            if (
                primitive_importance[i][0] in primitive_to_update
                and primitive_importance[i][1] < importance
            ):
                primitive_importance[i][1] = importance
        return primitive_importance

    def get_importance(
        self,
        dictionary,
        table_name,
        parent_table,
        variable,
        variable_importance_dictionary,
        primitive_importance,
        selection_variable="",
        selection_value="",
    ):
        """Get the importance measure of a variable by univariate analysis.

        :param dictionary: Khiops dictionary where the variable to estimate is.
        :param variable: Variable to estimate.
        :param variable_importance_dictionary: Variable importance dictionary to update.
        :param selection_variable: It trains with only the records such that the value of selection_variable is equal to selection_value, defaults to ""
        :type selection_variable: str, optional
        :param selection_value: See selection_variable option above, defaults to ""
        :type selection_value: str or int or float, optional
        :return variable_importance: Importance measure
        :rtype: float
        :return variable_importance_dictionary: Variable importance dictionary updated
        :rtype: dict
        """
        # Initialize importance measure
        variable_importance = 0
        max_aggregate = ""
        prefix = variable.name + "_" + table_name + "_" + selection_value + "_"
        # Set the variable to estimate to used -> only the variable and its associated table is used
        self.dictionary_domain.get_dictionary(dictionary.name).get_variable(
            variable.name
        ).used = True
        # Save dictionary if necessary
        self.dictionary_domain.export_khiops_dictionary_file(
            os.path.join(self.result_directory, "dico_" + prefix + ".kdic")
        )

        # Create variables (aggregates)
        if kh_version == "v10":
            train_reports_path, _ = kh.train_recoder(
                self.dictionary_domain,
                self.dictionary_name,
                self.root_table_path,
                self.target,
                self.result_directory,
                additional_data_tables=self.additional_table,
                results_prefix=prefix,
                max_constructed_variables=self.number_aggregate,
                max_trees=self.number_tree,
                construction_rules=self.construction_rules,
                selection_variable=selection_variable,
                selection_value=selection_value,
                keep_initial_categorical_variables=True,
                keep_initial_numerical_variables=True,
                informative_variables_only=False,
            )
        elif kh_version == "v11":
            train_reports_path, _ = kh.train_recoder(
                self.dictionary_domain,
                self.dictionary_name,
                self.root_table_path,
                self.target,
                os.path.join(self.result_directory, prefix + "AnalysisResults.khj"),
                additional_data_tables=self.additional_table,
                max_constructed_variables=self.number_aggregate,
                max_trees=self.number_tree,
                construction_rules=self.construction_rules,
                selection_variable=selection_variable,
                selection_value=selection_value,
                keep_initial_categorical_variables=True,
                keep_initial_numerical_variables=True,
                informative_variables_only=False,
            )
        # Update importance measure -> importance measure is the maximum Khiops level in the aggregate set.
        preparation_report = kh.read_analysis_results_file(
            train_reports_path
        ).preparation_report

        real_nb_agg = len(preparation_report.variables_statistics)
        for aggregate in preparation_report.variables_statistics:
            # if derivation rule is TableCount('Table') it is ignored
            # because it doesn't take into account the name of the variable

            if aggregate.derivation_rule == "TableCount(" + parent_table + ")":
                real_nb_agg -= 1
                continue

            # Get Variable importance -> maximum khiops level
            if self.exploration_type == "Variable" or self.exploration_type == "All":
                if aggregate.level > variable_importance:
                    variable_importance = aggregate.level
                    max_aggregate = aggregate.name
            # Get Primitive importance -> maximum khiops level
            if self.exploration_type == "Primitive" or self.exploration_type == "All":
                primitive_to_update = self.get_primitives_in_aggregate(
                    aggregate.derivation_rule
                )
                primitive_importance = self.update_importance_list_primitive(
                    primitive_importance, primitive_to_update, aggregate.level
                )

        # Set the variable estimated to unused
        self.dictionary_domain.get_dictionary(dictionary.name).get_variable(
            variable.name
        ).used = False

        # Update variable importance dictionary
        key_table = table_name
        key_var = variable.name
        var_type = variable.type
        var_level = variable_importance
        var_agg = max_aggregate
        var_nb_agg = real_nb_agg
        if key_table not in variable_importance_dictionary.keys():
            self.create_new_variable(
                variable_importance_dictionary,
                key_table,
                key_var,
                var_type,
                var_level,
                var_agg,
                var_nb_agg,
            )
        else:
            if key_var not in variable_importance_dictionary[key_table].keys():
                self.create_new_variable(
                    variable_importance_dictionary,
                    key_table,
                    key_var,
                    var_type,
                    var_level,
                    var_agg,
                    var_nb_agg,
                )
            else:
                if self.discretisation:
                    variable_importance_dictionary[key_table][key_var][
                        self.col_importances_list
                    ].append(var_level)
                    if (
                        var_level
                        > variable_importance_dictionary[key_table][key_var][
                            self.col_level
                        ]
                    ):
                        self.update_existed_variable(
                            variable_importance_dictionary,
                            key_table,
                            key_var,
                            var_type,
                            var_level,
                            var_agg,
                            var_nb_agg,
                        )

        return variable_importance_dictionary, primitive_importance

    def create_new_variable(
        self,
        variable_importance_dictionary,
        key_table,
        key_var,
        var_type,
        var_level,
        var_agg,
        var_nb_agg,
    ):
        variable_importance_dictionary[key_table][key_var] = {}
        variable_importance_dictionary = self.fill_line(
            variable_importance_dictionary,
            key_table,
            key_var,
            var_type,
            var_level,
            var_agg,
            var_nb_agg,
        )
        if self.discretisation:
            variable_importance_dictionary[key_table][key_var][
                self.col_importances_list
            ] = [var_level]
        return variable_importance_dictionary

    def update_existed_variable(
        self,
        variable_importance_dictionary,
        key_table,
        key_var,
        var_type,
        var_level,
        var_agg,
        var_nb_agg,
    ):
        variable_importance_dictionary = self.fill_line(
            variable_importance_dictionary,
            key_table,
            key_var,
            var_type,
            var_level,
            var_agg,
            var_nb_agg,
        )
        return variable_importance_dictionary

    def fill_line(
        self,
        variable_importance_dictionary,
        key_table,
        key_var,
        var_type,
        var_level,
        var_agg,
        var_nb_agg,
    ):
        variable_importance_dictionary[key_table][key_var][self.col_type] = var_type
        if self.discretisation:
            variable_importance_dictionary[key_table][key_var][self.col_level] = (
                var_level
            )
            variable_importance_dictionary[key_table][key_var][self.col_agg] = var_agg
        else:
            variable_importance_dictionary[key_table][key_var][
                self.col_level_no_discret
            ] = var_level
            variable_importance_dictionary[key_table][key_var][
                self.col_agg_no_discret
            ] = var_agg
        variable_importance_dictionary[key_table][key_var][self.col_nb_agg] = var_nb_agg
        return variable_importance_dictionary

    def add_variable(self, variable_to_add):
        """Adding khiops variable into dictionary domain

        :param variable_to_add: A list of variables to add into dictionary domain
        :type variable_to_add: list
        """
        for variable in variable_to_add:
            variable.used = False
            self.dictionary_domain.get_dictionary(self.dictionary_name).add_variable(
                variable
            )

    def remove_variable(self, variable_to_remove):
        """Remove khiops variable from dictionary domain

        :param variable_to_remove: A list of variables to remove from dictionary domain
        :type variable_to_remove: list
        """
        for variable in variable_to_remove:
            self.dictionary_domain.get_dictionary(self.dictionary_name).remove_variable(
                variable.name
            )

    def univariate_analysis(self):
        """
        Analyse variables by estimated a measure of importance for each variables using a
        n univariate analysis with discretisation.
        Create a  list of importance.
        """
        variable_importance_dictionary = self.init_variable_importance_dictionary()
        primitive_importance = self.init_importance_list_primitive()
        self.init_columns_names()
        nb_tables = 0

        # Get dictionary root name
        for dictionary in self.dictionary_domain.dictionaries:
            if dictionary.root:
                dictionary_root = dictionary.name
                break

        if self.discretisation:
            # Get count grouping variable to add into dictionary domain -> depending of the table

            variable_to_add = self.get_grouping_variable()

            # table schema analysis : number of table for each datatable ########## not used
            match_datatable_table_number_in_schema = {}
            # table schema analysis : construct the table dependency list
            for (
                table_name,
                dictionary_name,
            ) in self.match_name_dictionary_variable_name.items():
                number = 1
                for dictionary in self.dictionary_domain.dictionaries:
                    if dictionary_name == dictionary.name:
                        if not dictionary.root:
                            # construct the table dependency list
                            list_table_name = []
                            origine = dictionary.name
                            parent = self.match_dictionary_parent_dictionary[origine]
                            match_datatable_table_number_in_schema[table_name] = number
                            parent_table = table_name
                            list_table_name.append(parent_table)
                            # if parent is not root continue exploration
                            while parent != dictionary_root:
                                number += 1
                                origine = parent
                                parent = self.match_dictionary_parent_dictionary[
                                    origine
                                ]
                                match_datatable_table_number_in_schema[table_name] = (
                                    number
                                )
                                origine_table_name = get_key(
                                    self.match_name_dictionary_variable_name, origine
                                )
                                parent_table = origine_table_name
                                list_table_name.append(parent_table)
                            # copy the first discretisation to every dependent table

                            if len(list_table_name) > 1:
                                try:
                                    variable_to_add[list_table_name[0]]
                                except KeyError:
                                    variable_to_add[list_table_name[0]] = (
                                        variable_to_add[list_table_name[-1]]
                                    )

            # variable_to_add = self.get_grouping_variable(match_datatable_table_number_in_schema)
            # print(variable_to_add)

        # Analyse by table in match_name_dictionary_variable_name
        # With selection :
        #   match_name_dictionary_variable_name =
        #       {'Place': 'Place', 'Vehicles': 'Vehicle', 'UsersSelection': 'User'}
        # Snowflake schema :
        #   match_name_dictionary_variable_name =
        #       {'Place': 'Place', 'Vehicles': 'Vehicle', 'Users': 'User'}
        #   match_dictionary_parent_dictionary =
        #       {'Place': 'Accident', 'Accident': 'User', 'User': 'Vehicle'}
        for dictionary in self.dictionary_domain.dictionaries:
            if dictionary.root:
                dictionary_root = dictionary.name
                break

        for (
            table_name,
            dictionary_name,
        ) in self.match_name_dictionary_variable_name.items():
            print("table : " + table_name)
            for dictionary in self.dictionary_domain.dictionaries:
                if dictionary_name == dictionary.name:
                    if not dictionary.root:
                        # Set the table to analyse to used
                        origine = dictionary.name
                        parent = self.match_dictionary_parent_dictionary[origine]
                        self.dictionary_domain.get_dictionary(parent).get_variable(
                            table_name
                        ).used = True
                        parent_table = table_name  # to ignore the count on this table

                        # if parent is not root set the parent table to used
                        while parent != dictionary_root:
                            origine = parent
                            parent = self.match_dictionary_parent_dictionary[origine]
                            origine_table_name = get_key(
                                self.match_name_dictionary_variable_name, origine
                            )
                            self.dictionary_domain.get_dictionary(parent).get_variable(
                                origine_table_name
                            ).used = True
                            parent_table = (
                                origine_table_name  # to ignore the count on this table
                            )

                        variable_importance_dictionary[table_name] = {}

                        if self.discretisation:
                            # Add variable for the selection -> only for Table type table
                            if table_name in variable_to_add.keys():
                                self.add_variable(variable_to_add[table_name][:-1])
                                # Number of group to split instances
                                number_group = variable_to_add[table_name][-1]
                                for i in range(number_group):
                                    print(
                                        "discretization : group "
                                        + str(i + 1)
                                        + "/"
                                        + str(number_group)
                                    )
                                    categorical_variable = 0
                                    numerical_variable = 0
                                    date_variable = 0
                                    # Get importance measure for each variable of the table
                                    for variable in tqdm(dictionary.variables):
                                        # Get the number of variable type in tables
                                        if variable.type == "Categorical":
                                            categorical_variable += 1
                                        elif variable.type == "Numerical":
                                            numerical_variable += 1
                                        else:
                                            date_variable += 1
                                        if variable.name not in self.unused_variable:
                                            (
                                                variable_importance_dictionary,
                                                primitive_importance,
                                            ) = self.get_importance(
                                                dictionary,
                                                table_name,
                                                parent_table,
                                                variable,
                                                variable_importance_dictionary,
                                                primitive_importance,
                                                selection_variable=variable_to_add[
                                                    table_name
                                                ][0].name,
                                                selection_value="I" + str(i + 1),
                                            )

                                # Remove variable selection of the table
                                self.remove_variable(variable_to_add[table_name][:-1])
                            # get unique importance list of Entity type table
                            else:
                                categorical_variable = 0
                                numerical_variable = 0
                                date_variable = 0
                                for variable in tqdm(dictionary.variables):
                                    # Get the number of variable type in tables
                                    if variable.type == "Categorical":
                                        categorical_variable += 1
                                    elif variable.type == "Numerical":
                                        numerical_variable += 1
                                    else:
                                        date_variable += 1
                                    if variable.name not in self.unused_variable:
                                        (
                                            variable_importance_dictionary,
                                            primitive_importance,
                                        ) = self.get_importance(
                                            dictionary,
                                            table_name,
                                            parent_table,
                                            variable,
                                            variable_importance_dictionary,
                                            primitive_importance,
                                        )

                        else:
                            # without count effect reduction
                            categorical_variable = 0
                            numerical_variable = 0
                            date_variable = 0
                            # Get importance measure for each variable of the table
                            for variable in tqdm(dictionary.variables):
                                # Get the number of variable type in tables
                                if variable.type == "Categorical":
                                    categorical_variable += 1
                                elif variable.type == "Numerical":
                                    numerical_variable += 1
                                else:
                                    date_variable += 1
                                if variable.name not in self.unused_variable:
                                    # Get variable and/or primitive importance
                                    (
                                        variable_importance_dictionary,
                                        primitive_importance,
                                    ) = self.get_importance(
                                        dictionary,
                                        table_name,
                                        parent_table,
                                        variable,
                                        variable_importance_dictionary,
                                        primitive_importance,
                                    )

                        # Set the table to analyse to unused
                        origine = dictionary.name
                        parent = self.match_dictionary_parent_dictionary[origine]
                        self.dictionary_domain.get_dictionary(parent).get_variable(
                            table_name
                        ).used = False
                        # if parent is not root set the parent table to unused
                        while parent != dictionary_root:
                            origine = parent
                            parent = self.match_dictionary_parent_dictionary[origine]
                            origine_table_name = get_key(
                                self.match_name_dictionary_variable_name, origine
                            )
                            self.dictionary_domain.get_dictionary(parent).get_variable(
                                origine_table_name
                            ).used = False

                    else:
                        categorical_variable = 0
                        numerical_variable = 0
                        date_variable = 0
                        for variable in tqdm(dictionary.variables):
                            # Get the number of variable type in tables
                            if variable.type == "Categorical":
                                categorical_variable += 1
                            elif variable.type == "Numerical":
                                numerical_variable += 1
                            else:
                                date_variable += 1

                    self.table_exploration_array.loc[nb_tables] = [
                        table_name,
                        dictionary.root,
                        len(dictionary.variables),
                        categorical_variable,
                        numerical_variable,
                        date_variable,
                    ]
                    nb_tables += 1
                    break

        return (variable_importance_dictionary, primitive_importance)

    def variables_analysis(self):
        """Global function to estimate variable's importance"""
        # Get matching dictionary for variable table name and table name
        self.match_dictionary_name_variable_name()
        # Initialise dictionary domain -> all variable to unused
        self.initialize_variable_state()

        (
            variable_importance_dictionary,
            primitive_importance,
        ) = self.univariate_analysis()

        if self.exploration_type == "Variable" or self.exploration_type == "All":
            variable_exploration_file_name = "variable_exploration.json"
            variable_exploration_file = os.path.join(
                self.output_dir, variable_exploration_file_name
            )
            with open(variable_exploration_file, "w") as f:
                json.dump(variable_importance_dictionary, f, indent=2)

        if self.exploration_type == "Primitive" or self.exploration_type == "All":
            primitive_exploration_file_name = "primitive_exploration.json"
            primitive_exploration_file = os.path.join(
                self.output_dir, primitive_exploration_file_name
            )
            with open(primitive_exploration_file, "w") as f:
                json.dump(primitive_importance, f, indent=2)

        if self.exploration_type == "All":
            return variable_importance_dictionary, primitive_importance
        elif self.exploration_type == "Variable":
            return variable_importance_dictionary
        elif self.exploration_type == "Primitive":
            return primitive_importance
