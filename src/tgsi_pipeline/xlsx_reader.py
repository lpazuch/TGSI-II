from __future__ import annotations

import io
from pathlib import PurePosixPath
from zipfile import ZipFile
import xml.etree.ElementTree as ET


NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}
OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _excel_col_to_index(reference: str) -> int:
    letters = "".join(character for character in reference if character.isalpha())
    index = 0
    for character in letters:
        index = index * 26 + (ord(character.upper()) - 64)
    return index - 1


def _resolve_sheet_path(archive: ZipFile, preferred_sheet_name: str | None = None) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    sheets = workbook.findall("main:sheets/main:sheet", NS)
    if not sheets:
        raise ValueError("Workbook XLSX sem planilha.")

    selected_sheet = None
    if preferred_sheet_name:
        for sheet in sheets:
            if sheet.attrib.get("name") == preferred_sheet_name:
                selected_sheet = sheet
                break

    if selected_sheet is None:
        visible_sheets = [sheet for sheet in sheets if sheet.attrib.get("state") != "hidden"]
        selected_sheet = visible_sheets[0] if visible_sheets else sheets[0]

    rel_id = selected_sheet.attrib[f"{{{OFFICE_REL_NS}}}id"]
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))

    target = None
    for relation in relationships.findall("rel:Relationship", NS):
        if relation.attrib["Id"] == rel_id:
            target = relation.attrib["Target"]
            break

    if target is None:
        raise ValueError("Nao foi possivel localizar a planilha selecionada do XLSX.")

    if target.startswith("/"):
        return target.lstrip("/")
    return str(PurePosixPath("xl") / target)


def _load_shared_strings(archive: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for node in root.findall("main:si", NS):
        text_parts = [part.text or "" for part in node.findall(".//main:t", NS)]
        values.append("".join(text_parts))
    return values


def read_first_sheet_rows(data: bytes, *, preferred_sheet_name: str | None = None) -> list[list[str]]:
    with ZipFile(io.BytesIO(data)) as archive:
        shared_strings = _load_shared_strings(archive)
        sheet_path = _resolve_sheet_path(archive, preferred_sheet_name=preferred_sheet_name)
        sheet = ET.fromstring(archive.read(sheet_path))
        rows: list[list[str]] = []

        for row in sheet.findall(".//main:sheetData/main:row", NS):
            cells: dict[int, str] = {}
            max_index = -1
            for cell in row.findall("main:c", NS):
                ref = cell.attrib.get("r", "")
                cell_index = _excel_col_to_index(ref)
                max_index = max(max_index, cell_index)
                value_node = cell.find("main:v", NS)
                inline_node = cell.find("main:is/main:t", NS)
                cell_type = cell.attrib.get("t")

                value = ""
                if cell_type == "s" and value_node is not None:
                    value = shared_strings[int(value_node.text or "0")]
                elif inline_node is not None:
                    value = inline_node.text or ""
                elif value_node is not None and value_node.text is not None:
                    value = value_node.text

                cells[cell_index] = value

            values = [cells.get(index, "") for index in range(max_index + 1)]
            rows.append(values)
        return rows
