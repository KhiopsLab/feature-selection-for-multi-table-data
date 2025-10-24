import os
import pandas as pd
import sys
import json
from khiops import core as kh
from class_UnivariateMultitableAnalysis import UnivariateMultitableAnalysis

# Add the path to the parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "")))


class VariableSelectionStatistics:
    """Estimates the importance of secondary native variables in multi-table
    data on the target variable using a univariate approach with or without
    discretization.

    :param dictionary_file_path: Path of a Khiops dictionary file.
    :type dictionary_file_path: str
    :param exploration_type: Parameter to be analyze, 'All' for both variable and primitive, 'Variable' or 'Primitive' for only variable or primitive, defaults to 'Variable'.
    :type exploration_type: str
    :param count_effect_reduction: State of discretization, True is used, defaults to True.
    :type count_effect_reduction: bool
    :param results_dir: Path of the results directory, defaults to "results".
    :type results_dir: str, optional
    :param variable_exploration_name: Name of the JSON file generated during multi-table analysis of variables, defaults to "variable_exploration.json".
    :type variable_exploration_name: str
    :param primitive_exploration_name: Name of the JSON file generated during multi-table analysis of primitives, defaults to "primitive_exploration.json".
    :type primitive_exploration_name: str
    """

    def __init__(
        self,
        dictionary_file_path,
        exploration_type="Variable",
        count_effect_reduction=True,
        results_dir="",
        variable_exploration_name="variable_exploration.json",
        primitive_exploration_name="primitive_exploration.json",
    ):
        """
        Initialize class
        """
        self.dictionary_file_path = dictionary_file_path
        self.exploration_type = exploration_type
        self.discretization = count_effect_reduction  # State of discretization option
        self.results_dir = results_dir  # Results directory path
        self.variable_exploration_name = variable_exploration_name
        self.primitive_exploration_name = primitive_exploration_name

        # columns names for variables and primitives exploration
        self.init_columns()

        # create an instance of the UnivariateMultitableAnalysis class
        self.analysis = UnivariateMultitableAnalysis(
            dictionary_file_path=dictionary_file_path,
            dictionary_name="",
            data_table_path="",
            additional_data_tables="",
            target_variable="",
            output_khiops_dir=results_dir,
            results_dir=results_dir,
        )

        # initialize match_dictionary_name by calling the method
        self.analysis.match_dictionary_name()
        # access the dictionary match_table_name_dictionary_name
        self.match_table_name_dictionary_name = (
            self.analysis.match_table_name_dictionary_name
        )

    def init_columns(self):
        # columns names for variables and primitives exploration
        self.table = "table"
        self.variable = "variable"
        self.col_type = "type"
        self.col_level = "levelMT"
        self.col_agg = "levelMT aggregate"
        self.col_nb_agg = "real number of aggregates"
        self.col_importances_list = "importances list"
        self.col_primitive = "primitive"
        self.col_level_no_discret = "importance"
        self.col_agg_no_discret = "importance aggregate"

    def init_df_variable_exploration(self, nb_columns):
        """Initialize the pandas dataframe to save the variables exploration array

        :return: empty pandas dataframe
        :rtype: dataframe
        """
        # with count effect reduction
        # 5 column format : type, levelMT, levelMT aggregate, real number of aggregates, importances list
        if nb_columns == 5:
            df_variable_exploration = (
                pd.DataFrame(  # Pandas dataframe to save variables information
                    [],
                    columns=[
                        self.table,
                        self.variable,
                        self.col_type,
                        self.col_level,
                        self.col_agg,
                        self.col_nb_agg,
                        self.col_importances_list,
                    ],
                )
            )
        # without count effect reduction
        # 4 column format : type, importance, importance aggregate, real number of aggregates
        elif nb_columns == 4:
            self.discretization = False
            df_variable_exploration = (
                pd.DataFrame(  # Pandas dataframe to save variables information
                    [],
                    columns=[
                        self.table,
                        self.variable,
                        self.col_type,
                        self.col_level_no_discret,
                        self.col_agg_no_discret,
                        self.col_nb_agg,
                    ],
                )
            )
        else:
            print(
                "Warning : dictionary in json file doesn't respect the expected format"
            )

        return df_variable_exploration

    def write_to_markdown_report(
        self,
        df_variable_exploration="",
        df_primitive_exploration="",
    ):
        """Create a text file with univariate analysis results"""
        if self.exploration_type == "All" or self.exploration_type == "Variable":
            exploration_file_name = "variable_exploration_to_markdown.txt"
            exploration_file = open(
                os.path.join(self.results_dir, exploration_file_name), "w"
            )
            exploration_file.write(df_variable_exploration.to_markdown() + "\n")
            exploration_file.close()
            print(
                "Report file saved : "
                + os.path.join(self.results_dir, exploration_file_name)
            )

        if self.exploration_type == "All" or self.exploration_type == "Primitive":
            exploration_file_name = "primitive_exploration_to_markdown.txt"
            exploration_file = open(
                os.path.join(self.results_dir, exploration_file_name), "w"
            )
            exploration_file.write(df_primitive_exploration.to_markdown() + "\n")
            exploration_file.close()
            print(
                "Report file saved : "
                + os.path.join(self.results_dir, exploration_file_name)
            )

    def write_to_csv_report(
        self,
        df_variable_exploration="",
        df_primitive_exploration="",
    ):
        """Create a tabulate file with univariate analysis results"""

        if self.exploration_type == "All" or self.exploration_type == "Variable":
            exploration_file_name = "variable_exploration.txt"
            exploration_file = os.path.join(self.results_dir, exploration_file_name)
            df_variable_exploration.to_csv(exploration_file, sep="\t", index=False)
            print(f"Report file saved : {exploration_file}")

        if self.exploration_type == "All" or self.exploration_type == "Primitive":
            exploration_file_name = "primitive_exploration.txt"
            exploration_file = os.path.join(self.results_dir, exploration_file_name)
            df_primitive_exploration.to_csv(exploration_file, sep="\t", index=False)
            print(f"Report file saved : {exploration_file}")

    def read_data(self, variable_exploration_name, primitive_exploration_name):
        """
        read univariate multi-table analysis results
        """
        # default value to return if exploration_type = "Variable" or "Primitive"
        variable_importance_dictionary = {}
        primitive_importance = []

        # reading files if existing
        # reading variables report
        if self.exploration_type == "All" or self.exploration_type == "Variable":
            variable_exploration_file_name = variable_exploration_name
            variable_exploration_file = os.path.join(
                self.results_dir, variable_exploration_file_name
            )

            if os.path.exists(variable_exploration_file):
                with open(variable_exploration_file, "r") as f:
                    variable_importance_dictionary = json.load(f)
            else:
                print(
                    "file " + '"' + variable_exploration_file + '"' + " doesn't exist"
                )

        # reading primitives report
        if self.exploration_type == "All" or self.exploration_type == "Primitive":
            primitive_exploration_file_name = primitive_exploration_name
            primitive_exploration_file = os.path.join(
                self.results_dir, primitive_exploration_file_name
            )

            if os.path.exists(primitive_exploration_file):
                with open(primitive_exploration_file, "r") as f:
                    primitive_importance = json.load(f)
            else:
                print(
                    "file " + '"' + primitive_exploration_file + '"' + " doesn't exist"
                )

        return variable_importance_dictionary, primitive_importance

    def get_df_variable_importance(self, variable_importance_dictionary):
        """
        convert dict to pandas dataframe
        """
        for key1, value1 in variable_importance_dictionary.items():
            for key2, value2 in variable_importance_dictionary[key1].items():
                nb_columns = len(value2)
                break
            break

        df_variable_exploration = self.init_df_variable_exploration(nb_columns)
        i = 0

        if self.discretization:
            for key1, value1 in variable_importance_dictionary.items():
                for key2, value2 in variable_importance_dictionary[key1].items():
                    df_variable_exploration.loc[i] = [
                        key1,
                        key2,
                        value2[self.col_type],
                        value2[self.col_level],
                        value2[self.col_agg],
                        value2[self.col_nb_agg],
                        value2[self.col_importances_list],
                    ]
                    i += 1
            # sort dataframe by levelMT, variable, table
            df_variable_exploration = df_variable_exploration.sort_values(
                by=[self.col_level, self.variable, self.table],
                ascending=[False, True, True],
            )

        else:
            for key1, value1 in variable_importance_dictionary.items():
                for key2, value2 in variable_importance_dictionary[key1].items():
                    df_variable_exploration.loc[i] = [
                        key1,
                        key2,
                        value2[self.col_type],
                        value2[self.col_level_no_discret],
                        value2[self.col_agg_no_discret],
                        value2[self.col_nb_agg],
                    ]
                    i += 1
            # sort dataframe by importance, variable, table
            df_variable_exploration = df_variable_exploration.sort_values(
                by=[self.col_level_no_discret, self.variable, self.table],
                ascending=[False, True, True],
            )

        # add rank
        df_variable_exploration.reset_index(drop=True, inplace=True)
        df_variable_exploration["rank"] = df_variable_exploration.index + 1
        # move column 'rank' to the front
        df_variable_exploration = df_variable_exploration[
            ["rank"] + [col for col in df_variable_exploration.columns if col != "rank"]
        ]

        return df_variable_exploration

    def get_df_primitive_importance(self, primitive_importance):
        """
        convert list to pandas dataframe
        """
        df_primitive_exploration = pd.DataFrame(
            primitive_importance, columns=[self.col_primitive, self.col_level]
        )
        # sort dataframe by levelMT, variable, table
        df_primitive_exploration = df_primitive_exploration.sort_values(
            by=[self.col_level, self.col_primitive],
            ascending=[False, True],
        )

        # add rank
        df_primitive_exploration.reset_index(drop=True, inplace=True)
        df_primitive_exploration["rank"] = df_primitive_exploration.index + 1
        # move column 'rank' to the front
        df_primitive_exploration = df_primitive_exploration[
            ["rank"]
            + [col for col in df_primitive_exploration.columns if col != "rank"]
        ]
        return df_primitive_exploration

    def get_list_variable(self, df_variable_exploration):
        """
        list variables

        :param df_variable_exploration: univariate analysis results
        :type df_variable_exploration: pandas dataframe
        :return: list of variables
        :rtype: list
        """
        list_variables = df_variable_exploration[self.variable].tolist()
        print("List of variables analyzed :")
        print(list_variables)
        return list_variables

    def get_variable_number_zero_level(self, df_variable_exploration):
        """
        number of variables with level zero

        :param df_variable_exploration: univariate analysis results
        :type df_variable_exploration: pandas dataframe
        :return: number of variables with level zero
        :rtype: int
        """

        if self.discretization:
            nb_zero = len(
                df_variable_exploration[
                    df_variable_exploration[self.col_level] == 0
                ]
            )
        else:
            nb_zero = len(
                df_variable_exploration[
                    df_variable_exploration[self.col_level_no_discret] == 0
                ]
            )
        print("Number of variables with level zero : " + str(nb_zero))
        return nb_zero

    def filter_variables_in_dictionary(
        self, df_variable_exploration, nb_var_to_filter=None
    ):
        """
        filter the nb_var_to_filter variables with low levelMT in dictionary
        if nb_var_to_filter is None then filter all variables with levelMT zero

        :param df_variable_exploration: univariate analysis results
        :type df_variable_exploration: pandas dataframe
        :param nb_var_to_filter: number of variables to filter, default None
        :type nb_var_to_filter: int
        :return: khiops dictionary
        :rtype: dictionary_domain
        """
        # copy of the dictionary to create the dictionary with variables in Unused
        dictionary_domain_10 = kh.read_dictionary_file(self.dictionary_file_path)
        dictionary_domain_10_reduced = dictionary_domain_10.copy()
        mypath, myfile = os.path.split(self.dictionary_file_path)
        # indexes of the names of the columns of df_variable_exploration
        col_list = list(df_variable_exploration.columns)
        indice_table = col_list.index(self.table)
        indice_variable = col_list.index(self.variable)
        if self.discretization:
            indice_level = col_list.index(self.col_level)
        else:
            indice_level = col_list.index(self.col_level_no_discret)

        if nb_var_to_filter is None:
            # filter all variables with level zero
            print("\nFiltering all variables with level zero :")
            for row in df_variable_exploration.itertuples(index=False):
                level = row[indice_level]
                if level == 0:
                    var_to_unselect = row[indice_variable]
                    table = row[indice_table]
                    dico_name = self.match_table_name_dictionary_name[table]
                    dictionary_domain_10_reduced = self.unselect_var(
                        dictionary_domain_10_reduced, dico_name, var_to_unselect
                    )
        else:
            # filter the nb_var_to_filter variables with low level in dictionary
            print(f"\nFiltering {nb_var_to_filter} variables with low level :")
            i = 0
            # count zero level
            nb_level0 = 0
            # browse the lines starting from the end
            for _, row in df_variable_exploration[::-1].iterrows():
                level = row[indice_level]
                # count zero level
                if level == 0:
                    nb_level0 += 1
                if i < nb_var_to_filter:
                    i += 1
                    # unselect the variable
                    var_to_unselect = row[indice_variable]
                    table = row[indice_table]
                    dico_name = self.match_table_name_dictionary_name[table]
                    dictionary_domain_10_reduced = self.unselect_var(
                        dictionary_domain_10_reduced, dico_name, var_to_unselect
                    )

            # display a warning if all variables with a level 0 haven't been set to Unused
            if nb_var_to_filter < nb_level0:
                print(
                    "Warning : the number of variables specified for "
                    f"filtering ({nb_var_to_filter}) is less than the number "
                    f"of variables with a level 0 ({nb_level0}), "
                    "the filtered variables are taken starting from the end "
                    "(in alphabetical order)"
                )

        # writing dictionary
        if nb_var_to_filter is None:
            str_nb = "level_0"
        else:
            str_nb = "filter_" + str(nb_var_to_filter)
        dictionary_file_path_reduced = os.path.join(
            mypath, "reduced_dictionary_" + str_nb + ".kdic"
        )
        dictionary_domain_10_reduced.export_khiops_dictionary_file(
            dictionary_file_path_reduced
        )
        print(f"Dictionary saved : {dictionary_file_path_reduced}")
        # return dictionary_domain_10_reduced

    def unselect_var(self, dictionary_domain, dico_name, var_to_unselect):
        # search the variable and assign it to Unused
        flag = False
        for dico in dictionary_domain.dictionaries:
            if not dico.root:
                if dico.name == dico_name:
                    for var in dico.variables:
                        if var.name == var_to_unselect:
                            flag = True
                            if not dico.is_key_variable(var):
                                var.used = False
                            else:
                                print(
                                    f"The variable '{var_to_unselect}' "
                                    f"in the table '{dico_name}' "
                                    "is a key, it is not set to Unused"
                                )
                            break
        if not flag:
            print(
                "Warning : the variable "
                + '"'
                + var_to_unselect
                + '"'
                + " in "
                + '"'
                + dico_name
                + '"'
                + " dictionary doesn't exist in the dictionary"
            )
        return dictionary_domain

    def get_variables_analysis(self):
        """
        get variable importance and primitive importance json file
        convert into dataframes and print

        :return: variables importances
        :rtype: dataframe
        :return: primitives importances
        :rtype: dataframe
        """
        variable_importance_dictionary, primitive_importance = self.read_data(
            self.variable_exploration_name,
            self.primitive_exploration_name,
        )
        if variable_importance_dictionary != {}:
            df_variable_exploration = self.get_df_variable_importance(
                variable_importance_dictionary
            )
            print(df_variable_exploration)
            print("Variables results loaded")
        if primitive_importance != []:
            df_primitive_exploration = self.get_df_primitive_importance(
                primitive_importance
            )
            print(df_primitive_exploration)
            print("Primitive results loaded")

        if variable_importance_dictionary != {} and primitive_importance != []:
            return df_variable_exploration, df_primitive_exploration
        elif variable_importance_dictionary != {}:
            return df_variable_exploration
        elif primitive_importance != []:
            return df_primitive_exploration
