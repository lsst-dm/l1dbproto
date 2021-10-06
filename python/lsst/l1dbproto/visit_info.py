# This file is part of l1dbproto.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from typing import NamedTuple, Optional

from lsst.daf.base import DateTime


class VisitInfo(NamedTuple):
    """Information about a visit
    """
    visitId: int
    """Visit ID, serial integer"""

    visitTime: DateTime
    """Visit time"""

    lastObjectId: int
    """Highest existing DIAObject ID"""

    lastSourceId: int
    """Highest existing DIASource ID"""


class VisitInfoStore:
    """Trivial "database" to store and retrieve VisitInfo.

    Parameters
    ----------
    path : str
        Location of a file to store the visit data.
    """
    def __init__(self, path: str):
        self.path = path

    def lastVisit(self) -> Optional[VisitInfo]:
        """Returns last visit information.

        Returns
        -------
        visit : `VisitInfo` or `None`
            Last stored visit info or `None` if there was nothing stored yet.
        """
        try:
            with open(self.path, "r") as file:
                data = file.read()
        except Exception:
            # cannot open file
            return None
        words = data.split()
        if len(words) != 4:
            raise ValueError(f"Unexpected content of {self.path}: `{data}`")
        visitId = int(words[0])
        visitTime = DateTime(words[1], DateTime.TAI)
        lastObjectId = int(words[2])
        lastSourceId = int(words[3])
        return VisitInfo(visitId, visitTime, lastObjectId, lastSourceId)

    def saveVisit(self, visitId: int, visitTime: DateTime, lastObjectId: int, lastSourceId: int
                  ) -> None:
        """Store visit information.

        Parameters
        ----------
        visitId : `int`
            Visit identifier
        visitTime : `datetime.datetime`
            Visit timestamp.
        lastObjectId : `int`
            Highest existing DIAObject ID
        lastSourceId : `int`
            Highest existing DIASource ID
        """
        with open(self.path, "w") as file:
            isoTime = visitTime.toString(DateTime.TAI)
            file.write(f"{visitId} {isoTime} {lastObjectId} {lastSourceId}")
