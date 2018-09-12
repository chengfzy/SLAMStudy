#pragma once
#include <fstream>
#include <iostream>
#include <string>

/**
 * @brief Get section string
 * @param title Section title(name)
 * @return Section string
 */
std::string section(const std::string& title) {
    std::string fillStr((120 - title.size()) / 2, '=');
    return fillStr + " " + title + " " + fillStr;
}

/**
 * @brief Get sub section string
 * @param title Sub section title(name)
 * @return Sub section string
 */
std::string subSection(const std::string& title) {
    std::string fillStr((120 - title.size()) / 2, '-');
    return fillStr + " " + title + " " + fillStr;
}

/**
 * @brief Get paragraph string
 * @param title Paragraph title(name)
 * @return Paragraph string
 */
std::string paragraph(const std::string& title) {
    std::string fillStr((15 - title.size()) / 2, '=');
    return fillStr + " " + title + " " + fillStr;
}
