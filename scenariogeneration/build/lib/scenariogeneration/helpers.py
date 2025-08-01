"""
  scenariogeneration
  https://github.com/pyoscx/scenariogeneration

  This Source Code Form is subject to the terms of the Mozilla Public
  License, v. 2.0. If a copy of the MPL was not distributed with this
  file, You can obtain one at https://mozilla.org/MPL/2.0/.

  Copyright (c) 2022 The scenariogeneration Authors.

"""
import xml.etree.ElementTree as ET
from lxml import etree


def prettify(element, encoding=None):
    """Returns a bytes string representing a prettified version of an XML element.

    Parameters:
    ----------
        element (ET.Element): The XML element to prettify.
        encoding (str): The encoding to use for the output, defaults to 'utf-8'.
                        If None, then 'utf-8' will be used as default.

    Returns:
    ----------
        bytes: The prettified XML as bytes string with 4-space indentation.
    """
    if not isinstance(element, ET.Element):
        element = element.get_element()

    if encoding is None:
        encoding = "utf-8"

    # Define a 4-space indent string
    indent_str = "    "

    # Use the etree.Parser class from lxml to specify a custom parser
    parser = etree.XMLParser(remove_blank_text=True)

    # Convert the ElementTree element to an lxml etree form
    lxml_element = etree.fromstring(ET.tostring(element, encoding), parser=parser)

    # Now generate a 2-space indented pretty_print string (bytes type)
    pretty_print_bytes = etree.tostring(
        lxml_element, pretty_print=True, encoding=encoding
    )

    # Decode the bytes type pretty_print string to utf-8 encoded string, then replace 2-space indents with 4 spaces
    pretty_print_str = pretty_print_bytes.decode(encoding).replace("  ", indent_str)

    # Encode the string back into bytes type and return
    return pretty_print_str.encode(encoding)


def prettyprint(element, encoding=None):
    """returns the element prettyfied for writing to file or printing to the commandline

    Parameters
    ----------
        element (Element, or any generation class of scenariogeneration): element to print

        encoding (str): specify the output encoding
            Default: None (works best for printing in terminal on ubuntu atleast)

    """
    print(prettify(element, encoding=encoding))


def printToFile(element, filename, prettyprint=True, encoding="utf-8"):
    """prints the element to a xml file

    Parameters
    ----------
        element (Element): element to print

        filename (str): file to save to

        prettyprint (bool): pretty or "ugly" print

        encoding (str): specify the output encoding
            Default: 'utf-8'
    """
    if prettyprint:
        try:
            with open(filename, "wb") as file_handle:
                file_handle.write(prettify(element, encoding=encoding))
        except LookupError:
            print("%s is not a valid encoding option." % encoding)

    else:
        tree = ET.ElementTree(element)
        try:
            tree.write(filename, encoding=encoding)
        except LookupError:
            print("%s is not a valid encoding option." % encoding)


def enum2str(enum):
    """helper to create strings from enums that should contain space but have to have _

    Parameters
    ----------
        enum (Enum): a enum of pyodrx

    Returns
    -------
        enumstr (str): the enum as a string replacing _ with ' '

    """
    return enum.name.replace("_", " ")


def convert_bool(value):
    """Method to transform booleans to correct xml version (lower case)

    Parameter
    ---------
        value (bool): the boolean

    Return
    ------
        boolean (str)
    """
    if isinstance(value, str):
        if value == "true" or value == "1":
            return True
        elif value == "false" or value == "0":
            return False
        elif value[0] == "$":
            return value
        else:
            raise ValueError(
                value
                + "is not a valid type of float input to openscenario, if a string is used as a float value (parameter or expression), it should have a $ as the first char.."
            )

    if value:
        return "true"
    else:
        return "false"


def visualize_road(current_file_path):
    import sys
    import os
    import matplotlib.pyplot as plt

    if sys.platform.startswith("linux"):
        executable = "esmini_linux"
    else:
        executable = "exmini_mac"
    os.system(f"{os.path.join(os.path.dirname(__file__), f'../../{executable}/bin/odrplot')} \
                {os.path.splitext(current_file_path)[0]+'.xodr'} \
                {os.path.splitext(current_file_path)[0]+'.csv'}")
    
    # plot figure method 1 
    # print(os.path.splitext(current_file_path)[0]+'.png')
    # os.system(f"python {os.path.join(os.path.dirname(__file__), f'../../{executable}/EnvironmentSimulator/Applications/odrplot/xodr.py')} \
    #            {os.path.splitext(current_file_path)[0]+'.csv'} \
    #            {os.path.splitext(current_file_path)[0]+'.png'}")
    
    # plot figure method 2
    from crash_agent.opendrive_utils import plot_road
    plot_road(os.path.splitext(current_file_path)[0]+'.csv', os.path.splitext(current_file_path)[0]+'.png')

