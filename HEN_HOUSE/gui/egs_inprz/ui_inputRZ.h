/*

###############################################################################
#
#  EGSnrc user interface for egs_inprz
#  Copyright (C) 2015 National Research Council Canada
#
#  This file is part of EGSnrc.
#
#  EGSnrc is free software: you can redistribute it and/or modify it under
#  the terms of the GNU Affero General Public License as published by the
#  Free Software Foundation, either version 3 of the License, or (at your
#  option) any later version.
#
#  EGSnrc is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#  FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for
#  more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with EGSnrc. If not, see http://www.gnu.org/licenses/.
#
###############################################################################
#
#  Author:          Ernesto Mainegra-Hing, 2003
#
#  Contributors:    Frederic Tessier
#                   Blake Walters
#
###############################################################################

*/

/********************************************************************************
** Form generated from reading UI file 'inputRZ.ui'
**
** Created by: Qt User Interface Compiler version 5.15.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_INPUTRZ_H
#define UI_INPUTRZ_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_InputRZForm
{
public:
    QGridLayout *gridLayout_7;
    QLabel *bannerLabel;
    QHBoxLayout *horizontalLayout_17;
    QGroupBox *ButtonGroup4;
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout_15;
    QPushButton *ExecuteButton;
    QPushButton *PreviewRZButton;
    QPushButton *PrintButton;
    QPushButton *compileButton;
    QSpacerItem *horizontalSpacer_2;
    QGroupBox *ConfirmButtonGroup;
    QVBoxLayout *verticalLayout_31;
    QHBoxLayout *horizontalLayout_16;
    QPushButton *ApplyButton;
    QPushButton *CancelButton;
    QPushButton *HelpButton;
    QPushButton *AboutButton;
    QTabWidget *TabWidgetRZ;
    QWidget *GItab;
    QVBoxLayout *verticalLayout_30;
    QVBoxLayout *verticalLayout_28;
    QGridLayout *gridLayout;
    QHBoxLayout *hboxLayout;
    QGroupBox *UserCodeButtonGroup;
    QVBoxLayout *vboxLayout;
    QRadioButton *cavrzRadioButton;
    QRadioButton *dosrzRadioButton;
    QRadioButton *sprrzRadioButton;
    QRadioButton *flurzRadioButton;
    QGroupBox *TargetbuttonGroup;
    QVBoxLayout *vboxLayout1;
    QVBoxLayout *vboxLayout2;
    QRadioButton *OptradioButton;
    QRadioButton *NoOptradioButton;
    QRadioButton *DebugradioButton;
    QRadioButton *CleanradioButton;
    QGroupBox *UserCodeAreaButtonGroup;
    QVBoxLayout *vboxLayout3;
    QVBoxLayout *vboxLayout4;
    QRadioButton *HOMERadioButton;
    QRadioButton *HEN_HOUSERadioButton;
    QRadioButton *OtherAreaRadioButton;
    QGroupBox *PEGSDataAreaButtonGroup;
    QVBoxLayout *vboxLayout5;
    QVBoxLayout *vboxLayout6;
    QRadioButton *HOMEPegsRadioButton;
    QRadioButton *HEN_HOUSEPegsRadioButton;
    QRadioButton *OtherPegsAreaRadioButton;
    QRadioButton *PEGSlessRadioButton;
    QGroupBox *TitleGroupBox;
    QVBoxLayout *vboxLayout7;
    QLineEdit *TitleEdit;
    QLabel *theguyLabel;
    QVBoxLayout *verticalLayout_26;
    QGroupBox *InputFileGroupBox;
    QHBoxLayout *horizontalLayout_21;
    QHBoxLayout *horizontalLayout_18;
    QComboBox *InputFileComboBox;
    QPushButton *OpenFileButton;
    QGroupBox *pegs4GroupBox;
    QHBoxLayout *hboxLayout1;
    QHBoxLayout *hboxLayout2;
    QComboBox *pegs4ComboBox;
    QPushButton *Pegs4FileButton;
    QGroupBox *groupBox57;
    QHBoxLayout *hboxLayout3;
    QHBoxLayout *hboxLayout4;
    QComboBox *CONFcomboBox;
    QPushButton *SPECButton;
    QHBoxLayout *hboxLayout5;
    QPushButton *ConfigurationButton;
    QSpacerItem *spacer45;
    QPushButton *logfileButton;
    QWidget *IOtab;
    QHBoxLayout *hboxLayout6;
    QHBoxLayout *hboxLayout7;
    QVBoxLayout *vboxLayout8;
    QGroupBox *randNButtonGroup;
    QVBoxLayout *vboxLayout9;
    QVBoxLayout *vboxLayout10;
    QRadioButton *noRandRadioButton;
    QRadioButton *lastRandRadioButton;
    QRadioButton *allRandRadioButton;
    QGroupBox *DoseRegGroupBox;
    QVBoxLayout *vboxLayout11;
    QVBoxLayout *vboxLayout12;
    QHBoxLayout *hboxLayout8;
    QLabel *minPlnTextLabel;
    QSpacerItem *Spacer1;
    QSpinBox *minPlnSpinBox;
    QHBoxLayout *hboxLayout9;
    QLabel *maxPlnTextLabel;
    QSpacerItem *Spacer2;
    QSpinBox *maxPlnSpinBox;
    QHBoxLayout *hboxLayout10;
    QLabel *minCylTextLabel;
    QSpacerItem *Spacer3;
    QSpinBox *minCylSpinBox;
    QHBoxLayout *hboxLayout11;
    QLabel *maxCylTextLabel;
    QSpacerItem *Spacer4;
    QSpinBox *maxCylSpinBox;
    QFrame *Frame4;
    QVBoxLayout *vboxLayout13;
    QCheckBox *storeDataCheckBox;
    QSpacerItem *spacer41;
    QVBoxLayout *vboxLayout14;
    QFrame *Frame6;
    QVBoxLayout *vboxLayout15;
    QVBoxLayout *vboxLayout16;
    QHBoxLayout *hboxLayout12;
    QLabel *outoptTextLabel;
    QSpacerItem *Spacer78;
    QComboBox *outoptComboBox;
    QHBoxLayout *hboxLayout13;
    QLabel *etransportLabel;
    QSpacerItem *Spacer5;
    QComboBox *etransportComboBox;
    QHBoxLayout *hboxLayout14;
    QLabel *iwatchTextLabel;
    QSpacerItem *Spacer8;
    QComboBox *iwatchComboBox;
    QHBoxLayout *hboxLayout15;
    QLabel *restartTextLabel;
    QSpacerItem *Spacer7;
    QComboBox *irestartComboBox;
    QGroupBox *SPRRegGroupBox;
    QComboBox *sproutComboBox;
    QLabel *sproutTextLabel;
    QTableWidget *sproutTable;
    QVBoxLayout *vboxLayout17;
    QGroupBox *PrintFluSpecGroupBox;
    QComboBox *PrintFluSpeComboBox;
    QTableWidget *ListFluTable;
    QGroupBox *IPRIMARYGroupBox;
    QVBoxLayout *vboxLayout18;
    QComboBox *IPRIMARYComboBox;
    QGroupBox *sloteFluGroupBox;
    QLabel *sloteFluLabel;
    QLineEdit *sloteFluEdit;
    QTableWidget *sloteFluTable;
    QWidget *MCtab;
    QWidget *Layout46;
    QVBoxLayout *vboxLayout19;
    QHBoxLayout *hboxLayout16;
    QLabel *ncaseLabel;
    QSpacerItem *Spacer6_2;
    QLineEdit *ncaseEdit;
    QHBoxLayout *hboxLayout17;
    QLabel *maxCPULabel;
    QSpacerItem *Spacer9;
    QLineEdit *maxCPUEdit;
    QHBoxLayout *hboxLayout18;
    QLabel *statLabel;
    QSpacerItem *Spacer8_2;
    QLineEdit *statEdit;
    QGroupBox *kermaGroupBox;
    QCheckBox *kermaCheckBox;
    QGroupBox *randGroupBox;
    QVBoxLayout *vboxLayout20;
    QHBoxLayout *hboxLayout19;
    QLabel *rand1Label;
    QSpacerItem *Spacer4_2;
    QSpinBox *rand1SpinBox;
    QHBoxLayout *hboxLayout20;
    QLabel *rand2Label;
    QSpacerItem *Spacer5_2;
    QSpinBox *rand2SpinBox;
    QGroupBox *photregGroupBox;
    QHBoxLayout *hboxLayout21;
    QCheckBox *photregCheckBox;
    QWidget *layout157;
    QVBoxLayout *vboxLayout21;
    QGroupBox *ifullGroupBox;
    QVBoxLayout *vboxLayout22;
    QHBoxLayout *hboxLayout22;
    QLabel *ifullLabel;
    QSpacerItem *Spacer77;
    QComboBox *ifullComboBox;
    QGroupBox *phdGroupBox;
    QWidget *Layout30;
    QVBoxLayout *vboxLayout23;
    QHBoxLayout *hboxLayout23;
    QLabel *SLOTELabel;
    QSpacerItem *Spacer1_2;
    QLineEdit *SLOTEEdit;
    QHBoxLayout *hboxLayout24;
    QLabel *DELTAELabel;
    QSpacerItem *Spacer2_2;
    QLineEdit *DELTAEEdit;
    QTableWidget *phdTable;
    QWidget *Gtab;
    QGridLayout *gridLayout_24;
    QGridLayout *gridLayout_23;
    QHBoxLayout *horizontalLayout_20;
    QGroupBox *inputmethButtonGroup;
    QVBoxLayout *vboxLayout24;
    QVBoxLayout *vboxLayout25;
    QRadioButton *groupRadioButton;
    QRadioButton *individualRadioButton;
    QRadioButton *cavityRadioButton;
    QVBoxLayout *verticalLayout_3;
    QHBoxLayout *hboxLayout25;
    QHBoxLayout *hboxLayout26;
    QLabel *ZFaceLabel;
    QLineEdit *ZFaceEdit;
    QSpacerItem *horizontalSpacer_3;
    QLabel *CavityInfoLabel;
    QGroupBox *mediaGroupBox;
    QVBoxLayout *verticalLayout_36;
    QHBoxLayout *horizontalLayout;
    QLabel *mediaLabel;
    QSpacerItem *spacer42_2;
    QComboBox *mediaComboBox;
    QTableWidget *mediaTable;
    QHBoxLayout *horizontalLayout_12;
    QGroupBox *GroupBox53;
    QVBoxLayout *verticalLayout_33;
    QTableWidget *geometryTable;
    QGroupBox *GroupBox54;
    QVBoxLayout *verticalLayout_34;
    QTableWidget *cylTable;
    QWidget *CItab;
    QWidget *layoutWidget;
    QHBoxLayout *horizontalLayout_6;
    QVBoxLayout *verticalLayout_7;
    QGroupBox *cavityGroupBox;
    QHBoxLayout *horizontalLayout_4;
    QHBoxLayout *horizontalLayout_3;
    QVBoxLayout *verticalLayout_6;
    QSpacerItem *verticalSpacer;
    QGroupBox *ButtonGroup25;
    QVBoxLayout *verticalLayout_4;
    QRadioButton *thimbleRadioButton;
    QRadioButton *parallelplateRadioButton;
    QSpacerItem *verticalSpacer_2;
    QVBoxLayout *verticalLayout_5;
    QHBoxLayout *hboxLayout27;
    QLabel *wallLabel;
    QLineEdit *wallthickEdit;
    QHBoxLayout *hboxLayout28;
    QLabel *cavradLabel;
    QLineEdit *cavradEdit;
    QHBoxLayout *hboxLayout29;
    QLabel *cavlenLabel;
    QLineEdit *cavlenEdit;
    QHBoxLayout *hboxLayout30;
    QLabel *electradLabel;
    QLineEdit *electradEdit;
    QHBoxLayout *hboxLayout31;
    QLabel *wallmaterialLabel;
    QComboBox *wallmaterialComboBox;
    QHBoxLayout *hboxLayout32;
    QLabel *electrmatLabel;
    QComboBox *electrmatComboBox;
    QSpacerItem *verticalSpacer_3;
    QGroupBox *gr_indGroupBox;
    QHBoxLayout *horizontalLayout_5;
    QGroupBox *GroupBox56;
    QTableWidget *cavTable;
    QWidget *MDTab;
    QVBoxLayout *verticalLayout_52;
    QVBoxLayout *verticalLayout_51;
    QHBoxLayout *horizontalLayout_28;
    QGroupBox *ELGroupBox;
    QVBoxLayout *verticalLayout_47;
    QHBoxLayout *horizontalLayout_7;
    QVBoxLayout *vboxLayout26;
    QHBoxLayout *hboxLayout33;
    QLabel *AELabel;
    QLineEdit *AEEdit;
    QHBoxLayout *hboxLayout34;
    QLabel *APLabel;
    QLineEdit *APEdit;
    QVBoxLayout *vboxLayout27;
    QHBoxLayout *hboxLayout35;
    QLabel *UELabel;
    QLineEdit *UEEdit;
    QHBoxLayout *hboxLayout36;
    QLabel *UPLabel;
    QLineEdit *UPEdit;
    QGroupBox *MDGroupBox;
    QVBoxLayout *verticalLayout_48;
    QHBoxLayout *hboxLayout37;
    QLineEdit *MDFEdit;
    QPushButton *MDFileButton;
    QGroupBox *inpmediaGroupBox;
    QVBoxLayout *verticalLayout_50;
    QVBoxLayout *verticalLayout_49;
    QHBoxLayout *horizontalLayout_24;
    QGroupBox *compositionGroupBox;
    QVBoxLayout *verticalLayout_8;
    QTableWidget *pz_or_rhozTable;
    QVBoxLayout *verticalLayout_46;
    QGroupBox *medNameGroupBox;
    QVBoxLayout *verticalLayout_9;
    QComboBox *inpmediumComboBox;
    QHBoxLayout *horizontalLayout_19;
    QGroupBox *medTypeGroupBox;
    QVBoxLayout *verticalLayout_37;
    QComboBox *medTypeComboBox;
    QGroupBox *rhoGroupBox;
    QVBoxLayout *verticalLayout_44;
    QHBoxLayout *horizontalLayout_8;
    QLineEdit *rhoEdit;
    QComboBox *rhoScaleComboBox;
    QGroupBox *medOptionsGroupBox;
    QVBoxLayout *verticalLayout_45;
    QVBoxLayout *verticalLayout_35;
    QCheckBox *DCcheckBox;
    QCheckBox *ICRUradCheckBox;
    QHBoxLayout *horizontalLayout_10;
    QCheckBox *isGasCheckBox;
    QSpacerItem *horizontalSpacer_5;
    QLabel *gaspLabel;
    QLineEdit *gaspEdit;
    QLabel *gaspUnits;
    QGroupBox *DFgroupBox;
    QVBoxLayout *verticalLayout_10;
    QHBoxLayout *horizontalLayout_9;
    QLabel *DFSearchLabel;
    QComboBox *DFSearchComboBox;
    QHBoxLayout *hboxLayout38;
    QLineEdit *DFEdit;
    QHBoxLayout *hboxLayout39;
    QPushButton *DFBrowse;
    QWidget *SItab;
    QGridLayout *gridLayout_4;
    QHBoxLayout *horizontalLayout_11;
    QGridLayout *gridLayout_3;
    QVBoxLayout *verticalLayout_13;
    QGroupBox *in_particleButtonGroup;
    QGridLayout *gridLayout_2;
    QRadioButton *eRadioButton;
    QRadioButton *phRadioButton;
    QRadioButton *pRadioButton;
    QRadioButton *chargedRadioButton;
    QRadioButton *allRadioButton;
    QGroupBox *energyButtonGroup;
    QGridLayout *gridLayout1;
    QRadioButton *monoenergeticRadioButton;
    QRadioButton *spectrumRadioButton;
    QVBoxLayout *verticalLayout_16;
    QGroupBox *IniEgroupBox;
    QVBoxLayout *verticalLayout_15;
    QLineEdit *ini_energyEdit;
    QGroupBox *specfnameGroupBox;
    QVBoxLayout *verticalLayout_17;
    QHBoxLayout *hboxLayout40;
    QComboBox *specfnameComboBox;
    QPushButton *specfnameButton;
    QGroupBox *ioutspGroupBox;
    QFormLayout *formLayout;
    QCheckBox *ioutspCheckBox;
    QVBoxLayout *verticalLayout_14;
    QVBoxLayout *verticalLayout_12;
    QHBoxLayout *hboxLayout41;
    QLabel *sourceLabel;
    QComboBox *sourceComboBox;
    QGroupBox *sourceoptionsGroupBox;
    QVBoxLayout *verticalLayout_11;
    QVBoxLayout *vboxLayout28;
    QHBoxLayout *hboxLayout42;
    QLabel *temp1Label;
    QSpacerItem *Spacer1_4;
    QLineEdit *temp1Edit;
    QHBoxLayout *hboxLayout43;
    QLabel *temp2Label;
    QSpacerItem *Spacer2_4;
    QLineEdit *temp2Edit;
    QHBoxLayout *hboxLayout44;
    QLabel *temp3Label;
    QSpacerItem *Spacer3_3;
    QLineEdit *temp3Edit;
    QHBoxLayout *hboxLayout45;
    QLabel *temp4Label;
    QSpacerItem *Spacer4_4;
    QLineEdit *temp4Edit;
    QComboBox *imodeComboBox;
    QHBoxLayout *hboxLayout46;
    QLabel *temp5Label;
    QSpacerItem *Spacer36;
    QLineEdit *temp5Edit;
    QHBoxLayout *hboxLayout47;
    QLabel *temp6Label;
    QSpacerItem *Spacer37_2;
    QLineEdit *temp6Edit;
    QHBoxLayout *hboxLayout48;
    QLabel *temp7Label;
    QSpacerItem *Spacer38_2;
    QLineEdit *temp7Edit;
    QSpacerItem *verticalSpacer_5;
    QGroupBox *phasespaceGroupBox;
    QVBoxLayout *verticalLayout_18;
    QHBoxLayout *hboxLayout49;
    QComboBox *phasespaceComboBox;
    QPushButton *phasespacePushButton;
    QGroupBox *src20GroupBox;
    QGridLayout *gridLayout_5;
    QGroupBox *srcinpmodeButtonGroup;
    QHBoxLayout *hboxLayout50;
    QHBoxLayout *hboxLayout51;
    QRadioButton *localRadioButton;
    QRadioButton *externalRadioButton;
    QSpacerItem *spacer56_3;
    QTableWidget *raddistTable;
    QSpacerItem *spacer57_2;
    QVBoxLayout *vboxLayout29;
    QGroupBox *raddistfnameGroupBox;
    QHBoxLayout *hboxLayout52;
    QHBoxLayout *hboxLayout53;
    QComboBox *raddistfnameComboBox;
    QPushButton *raddistfnameButton;
    QGroupBox *ioutrdistGroupBox;
    QHBoxLayout *hboxLayout54;
    QCheckBox *ioutrdistCheckBox;
    QWidget *MCTPtab;
    QGridLayout *gridLayout_18;
    QGridLayout *gridLayout_17;
    QHBoxLayout *hboxLayout55;
    QVBoxLayout *vboxLayout30;
    QGroupBox *PairAngSamplingGroupBox;
    QHBoxLayout *hboxLayout56;
    QComboBox *PairAngSamplingComboBox;
    QGroupBox *BremsAngSamplingGroupBox;
    QHBoxLayout *hboxLayout57;
    QComboBox *BremsAngSamplingComboBox;
    QGroupBox *BremsXSectionGroupBox;
    QHBoxLayout *hboxLayout58;
    QComboBox *BremsXSectionComboBox;
    QVBoxLayout *vboxLayout31;
    QGroupBox *estep_algorithmGroupBox;
    QHBoxLayout *hboxLayout59;
    QComboBox *estep_algorithmComboBox;
    QGroupBox *EIIgroupBox;
    QVBoxLayout *vboxLayout32;
    QComboBox *EIIcomboBox;
    QGroupBox *BCAGroupBox;
    QHBoxLayout *hboxLayout60;
    QComboBox *BCAComboBox;
    QVBoxLayout *verticalLayout_20;
    QGroupBox *GECUTGroupBox;
    QGridLayout *gridLayout_8;
    QLineEdit *GECUTEdit;
    QGroupBox *GPCUTGroupBox;
    QGridLayout *gridLayout_15;
    QLineEdit *GPCUTEdit;
    QGroupBox *GSMAXGroupBox;
    QGridLayout *gridLayout_14;
    QLineEdit *GSMAXEdit;
    QGridLayout *gridLayout_16;
    QGridLayout *gridLayout_13;
    QGroupBox *RayleighgroupBox;
    QGridLayout *gridLayout_9;
    QComboBox *RayleighcomboBox;
    QGroupBox *PEgroupBox;
    QGridLayout *gridLayout_11;
    QComboBox *PEcomboBox;
    QGroupBox *BoundComptongroupBox;
    QGridLayout *gridLayout_6;
    QComboBox *BoundComptoncomboBox;
    QGroupBox *RelaxationgroupBox;
    QGridLayout *gridLayout_10;
    QComboBox *RelaxationcomboBox;
    QVBoxLayout *verticalLayout_19;
    QCheckBox *ECUTCheckBox;
    QCheckBox *PCUTCheckBox;
    QCheckBox *SMAXCheckBox;
    QVBoxLayout *vboxLayout33;
    QGroupBox *photonXSectiongroupBox;
    QVBoxLayout *vboxLayout34;
    QComboBox *photonXSectioncomboBox;
    QCheckBox *photonXSectionOutCheckBox;
    QHBoxLayout *horizontalLayout_27;
    QGroupBox *customFFgroupBox;
    QGridLayout *gridLayout_12;
    QTableWidget *customFFTable;
    QVBoxLayout *verticalLayout_27;
    QVBoxLayout *vboxLayout35;
    QVBoxLayout *vboxLayout36;
    QHBoxLayout *hboxLayout61;
    QLabel *ESTEPELabel;
    QLineEdit *ESTEPEEdit;
    QHBoxLayout *hboxLayout62;
    QLabel *XImaxLabel;
    QLineEdit *XImaxEdit;
    QHBoxLayout *hboxLayout63;
    QLabel *SkinDepthLabel;
    QLineEdit *SkinDepthEdit;
    QCheckBox *SpinCheckBox;
    QSpacerItem *verticalSpacer_4;
    QWidget *VRtab;
    QWidget *layoutWidget3;
    QHBoxLayout *horizontalLayout_22;
    QVBoxLayout *vboxLayout37;
    QGroupBox *PhotonForcingGroupBox;
    QVBoxLayout *vboxLayout38;
    QVBoxLayout *vboxLayout39;
    QCheckBox *PhotonForcingCheckBox;
    QHBoxLayout *hboxLayout64;
    QVBoxLayout *vboxLayout40;
    QLabel *StartForcingLabel;
    QLabel *StopForcingLabel;
    QVBoxLayout *vboxLayout41;
    QSpinBox *StartForcingSpinBox;
    QSpinBox *StopForcingSpinBox;
    QGroupBox *ExpTrafoCGroupBox;
    QHBoxLayout *hboxLayout65;
    QHBoxLayout *hboxLayout66;
    QLabel *ExpTrafoCLabel;
    QLineEdit *ExpTrafoCEdit;
    QGroupBox *RRGroupBox;
    QVBoxLayout *vboxLayout42;
    QVBoxLayout *vboxLayout43;
    QHBoxLayout *hboxLayout67;
    QLabel *RRDepthLabel;
    QLineEdit *RRDepthEdit;
    QHBoxLayout *hboxLayout68;
    QLabel *RRFractionLabel;
    QLineEdit *RRFractionEdit;
    QGroupBox *photonSplitGroupBox;
    QHBoxLayout *hboxLayout69;
    QHBoxLayout *hboxLayout70;
    QSpinBox *photonSplitSpinBox;
    QLabel *photonSplitLabel;
    QVBoxLayout *vboxLayout44;
    QGroupBox *CSEnhancementGroupBox;
    QWidget *layoutWidget4;
    QVBoxLayout *verticalLayout_2;
    QHBoxLayout *horizontalLayout_2;
    QLabel *TextLabel8;
    QSpinBox *CSEnhancementSpinBox;
    QVBoxLayout *vboxLayout45;
    QVBoxLayout *vboxLayout46;
    QLabel *CSEnhancement_RegionsLabel1;
    QLabel *CSEnhancement_RegionsLabel2;
    QHBoxLayout *hboxLayout71;
    QTableWidget *CSEnhancementTable;
    QVBoxLayout *vboxLayout47;
    QGroupBox *ESAVEINGroupBox;
    QVBoxLayout *vboxLayout48;
    QVBoxLayout *vboxLayout49;
    QCheckBox *eRangeRejCheckBox;
    QHBoxLayout *hboxLayout72;
    QLabel *ESAVEINLabel;
    QSpacerItem *Spacer74;
    QLineEdit *ESAVEINEdit;
    QGroupBox *BremsSplitGroupBox;
    QVBoxLayout *vboxLayout50;
    QVBoxLayout *vboxLayout51;
    QHBoxLayout *hboxLayout73;
    QSpinBox *BremsSplitSpinBox;
    QLabel *BremsSplitTextLabel;
    QCheckBox *BremsSplitCheckBox;
    QCheckBox *ChargedPartRRCheckBox;
    QWidget *MCTPRegTab;
    QWidget *layoutWidget5;
    QVBoxLayout *verticalLayout_43;
    QHBoxLayout *horizontalLayout_26;
    QGroupBox *PCUTGroupBox;
    QVBoxLayout *verticalLayout_40;
    QTableWidget *PCUTTable;
    QGroupBox *ECUTGroupBox;
    QVBoxLayout *verticalLayout_41;
    QTableWidget *ECUTTable;
    QGroupBox *SMAXGroupBox;
    QVBoxLayout *verticalLayout_42;
    QTableWidget *SMAXTable;
    QHBoxLayout *horizontalLayout_25;
    QGroupBox *BoundComptonGroupBox;
    QVBoxLayout *verticalLayout_39;
    QTableWidget *BoundComptonTable;
    QGroupBox *RelaxationsGroupBox;
    QVBoxLayout *verticalLayout_38;
    QTableWidget *RelaxationsTable;
    QGroupBox *PEAngSamGroupBox;
    QVBoxLayout *verticalLayout_29;
    QTableWidget *PEAngSamplingTable;
    QGroupBox *RayleighGroupBox;
    QVBoxLayout *verticalLayout_32;
    QTableWidget *RayleighTable;
    QWidget *PLOTtab;
    QGridLayout *gridLayout_22;
    QVBoxLayout *verticalLayout_25;
    QCheckBox *plotCheckBox;
    QGroupBox *PlotGroupBox;
    QFormLayout *formLayout_2;
    QHBoxLayout *horizontalLayout_13;
    QVBoxLayout *verticalLayout_24;
    QVBoxLayout *verticalLayout_21;
    QCheckBox *LinePrnOutCheckBox;
    QCheckBox *ExtPlotOutCheckBox;
    QGroupBox *ExternalPlotTypeGroupBox;
    QGridLayout *gridLayout_20;
    QComboBox *ExternalPlotTypeComboBox;
    QGroupBox *PlotRegionsGroupBox;
    QHBoxLayout *horizontalLayout_23;
    QTableWidget *PlotRegionsTable;
    QSpacerItem *horizontalSpacer_4;
    QGridLayout *gridLayout_21;
    QVBoxLayout *verticalLayout_22;
    QCheckBox *eminusPlotCheckBox;
    QCheckBox *eplusPlotCheckBox;
    QVBoxLayout *verticalLayout_23;
    QCheckBox *gammaPlotCheckBox;
    QCheckBox *ePlotCheckBox;
    QGroupBox *DrawFluPlotsGroupBox;
    QGridLayout *gridLayout_19;
    QComboBox *DrawFluPlotsComboBox;
    QGroupBox *SpecPlotGroupBox;
    QHBoxLayout *horizontalLayout_14;
    QTableWidget *SpecPlotTable;

    void setupUi(QWidget *InputRZForm)
    {
        if (InputRZForm->objectName().isEmpty())
            InputRZForm->setObjectName(QString::fromUtf8("InputRZForm"));
        InputRZForm->resize(1024, 780);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(InputRZForm->sizePolicy().hasHeightForWidth());
        InputRZForm->setSizePolicy(sizePolicy);
        InputRZForm->setMinimumSize(QSize(900, 780));
        InputRZForm->setMaximumSize(QSize(1024, 820));
        InputRZForm->setMouseTracking(false);
        gridLayout_7 = new QGridLayout(InputRZForm);
        gridLayout_7->setSpacing(6);
        gridLayout_7->setContentsMargins(11, 11, 11, 11);
        gridLayout_7->setObjectName(QString::fromUtf8("gridLayout_7"));
        bannerLabel = new QLabel(InputRZForm);
        bannerLabel->setObjectName(QString::fromUtf8("bannerLabel"));
        QSizePolicy sizePolicy1(QSizePolicy::Minimum, QSizePolicy::Minimum);
        sizePolicy1.setHorizontalStretch(232);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(bannerLabel->sizePolicy().hasHeightForWidth());
        bannerLabel->setSizePolicy(sizePolicy1);
        bannerLabel->setMinimumSize(QSize(860, 88));
        bannerLabel->setMaximumSize(QSize(800, 88));
        bannerLabel->setFrameShape(QFrame::Box);
        bannerLabel->setFrameShadow(QFrame::Plain);
        bannerLabel->setLineWidth(0);
        bannerLabel->setScaledContents(true);
        bannerLabel->setAlignment(Qt::AlignVCenter);
        bannerLabel->setWordWrap(false);

        gridLayout_7->addWidget(bannerLabel, 0, 0, 1, 1);

        horizontalLayout_17 = new QHBoxLayout();
        horizontalLayout_17->setSpacing(6);
        horizontalLayout_17->setObjectName(QString::fromUtf8("horizontalLayout_17"));
        ButtonGroup4 = new QGroupBox(InputRZForm);
        ButtonGroup4->setObjectName(QString::fromUtf8("ButtonGroup4"));
        QSizePolicy sizePolicy2(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(ButtonGroup4->sizePolicy().hasHeightForWidth());
        ButtonGroup4->setSizePolicy(sizePolicy2);
        verticalLayout = new QVBoxLayout(ButtonGroup4);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        horizontalLayout_15 = new QHBoxLayout();
        horizontalLayout_15->setSpacing(6);
        horizontalLayout_15->setObjectName(QString::fromUtf8("horizontalLayout_15"));
        ExecuteButton = new QPushButton(ButtonGroup4);
        ExecuteButton->setObjectName(QString::fromUtf8("ExecuteButton"));
        ExecuteButton->setEnabled(false);
        QSizePolicy sizePolicy3(QSizePolicy::Minimum, QSizePolicy::Fixed);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(ExecuteButton->sizePolicy().hasHeightForWidth());
        ExecuteButton->setSizePolicy(sizePolicy3);
        ExecuteButton->setChecked(false);
        ExecuteButton->setAutoDefault(false);

        horizontalLayout_15->addWidget(ExecuteButton);

        PreviewRZButton = new QPushButton(ButtonGroup4);
        PreviewRZButton->setObjectName(QString::fromUtf8("PreviewRZButton"));
        PreviewRZButton->setEnabled(false);
        PreviewRZButton->setAutoDefault(false);

        horizontalLayout_15->addWidget(PreviewRZButton);

        PrintButton = new QPushButton(ButtonGroup4);
        PrintButton->setObjectName(QString::fromUtf8("PrintButton"));
        PrintButton->setEnabled(false);
        PrintButton->setAutoDefault(false);

        horizontalLayout_15->addWidget(PrintButton);

        compileButton = new QPushButton(ButtonGroup4);
        compileButton->setObjectName(QString::fromUtf8("compileButton"));
        compileButton->setEnabled(true);
        compileButton->setAutoDefault(false);

        horizontalLayout_15->addWidget(compileButton);


        verticalLayout->addLayout(horizontalLayout_15);


        horizontalLayout_17->addWidget(ButtonGroup4);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_17->addItem(horizontalSpacer_2);

        ConfirmButtonGroup = new QGroupBox(InputRZForm);
        ConfirmButtonGroup->setObjectName(QString::fromUtf8("ConfirmButtonGroup"));
        verticalLayout_31 = new QVBoxLayout(ConfirmButtonGroup);
        verticalLayout_31->setSpacing(6);
        verticalLayout_31->setContentsMargins(11, 11, 11, 11);
        verticalLayout_31->setObjectName(QString::fromUtf8("verticalLayout_31"));
        horizontalLayout_16 = new QHBoxLayout();
        horizontalLayout_16->setSpacing(6);
        horizontalLayout_16->setObjectName(QString::fromUtf8("horizontalLayout_16"));
        ApplyButton = new QPushButton(ConfirmButtonGroup);
        ApplyButton->setObjectName(QString::fromUtf8("ApplyButton"));
        ApplyButton->setAutoDefault(false);

        horizontalLayout_16->addWidget(ApplyButton);

        CancelButton = new QPushButton(ConfirmButtonGroup);
        CancelButton->setObjectName(QString::fromUtf8("CancelButton"));
        CancelButton->setAutoDefault(false);

        horizontalLayout_16->addWidget(CancelButton);

        HelpButton = new QPushButton(ConfirmButtonGroup);
        HelpButton->setObjectName(QString::fromUtf8("HelpButton"));
        HelpButton->setAutoDefault(false);

        horizontalLayout_16->addWidget(HelpButton);

        AboutButton = new QPushButton(ConfirmButtonGroup);
        AboutButton->setObjectName(QString::fromUtf8("AboutButton"));

        horizontalLayout_16->addWidget(AboutButton);


        verticalLayout_31->addLayout(horizontalLayout_16);


        horizontalLayout_17->addWidget(ConfirmButtonGroup);


        gridLayout_7->addLayout(horizontalLayout_17, 2, 0, 1, 1);

        TabWidgetRZ = new QTabWidget(InputRZForm);
        TabWidgetRZ->setObjectName(QString::fromUtf8("TabWidgetRZ"));
        TabWidgetRZ->setEnabled(true);
        sizePolicy2.setHeightForWidth(TabWidgetRZ->sizePolicy().hasHeightForWidth());
        TabWidgetRZ->setSizePolicy(sizePolicy2);
        TabWidgetRZ->setMinimumSize(QSize(0, 0));
        TabWidgetRZ->setMaximumSize(QSize(30000, 30000));
        QFont font;
        font.setPointSize(13);
        font.setBold(false);
        font.setItalic(false);
        font.setWeight(50);
        TabWidgetRZ->setFont(font);
        TabWidgetRZ->setMouseTracking(false);
        TabWidgetRZ->setTabPosition(QTabWidget::North);
        TabWidgetRZ->setTabShape(QTabWidget::Rounded);
        TabWidgetRZ->setIconSize(QSize(11, 16));
        TabWidgetRZ->setElideMode(Qt::ElideNone);
        TabWidgetRZ->setUsesScrollButtons(true);
        TabWidgetRZ->setDocumentMode(false);
        TabWidgetRZ->setTabsClosable(false);
        TabWidgetRZ->setMovable(false);
        GItab = new QWidget();
        GItab->setObjectName(QString::fromUtf8("GItab"));
        verticalLayout_30 = new QVBoxLayout(GItab);
        verticalLayout_30->setSpacing(6);
        verticalLayout_30->setContentsMargins(11, 11, 11, 11);
        verticalLayout_30->setObjectName(QString::fromUtf8("verticalLayout_30"));
        verticalLayout_28 = new QVBoxLayout();
        verticalLayout_28->setSpacing(6);
        verticalLayout_28->setObjectName(QString::fromUtf8("verticalLayout_28"));
        gridLayout = new QGridLayout();
        gridLayout->setSpacing(6);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        hboxLayout = new QHBoxLayout();
        hboxLayout->setSpacing(6);
        hboxLayout->setObjectName(QString::fromUtf8("hboxLayout"));
        UserCodeButtonGroup = new QGroupBox(GItab);
        UserCodeButtonGroup->setObjectName(QString::fromUtf8("UserCodeButtonGroup"));
        vboxLayout = new QVBoxLayout(UserCodeButtonGroup);
        vboxLayout->setSpacing(6);
        vboxLayout->setContentsMargins(11, 11, 11, 11);
        vboxLayout->setObjectName(QString::fromUtf8("vboxLayout"));
        cavrzRadioButton = new QRadioButton(UserCodeButtonGroup);
        cavrzRadioButton->setObjectName(QString::fromUtf8("cavrzRadioButton"));
        cavrzRadioButton->setChecked(true);

        vboxLayout->addWidget(cavrzRadioButton);

        dosrzRadioButton = new QRadioButton(UserCodeButtonGroup);
        dosrzRadioButton->setObjectName(QString::fromUtf8("dosrzRadioButton"));

        vboxLayout->addWidget(dosrzRadioButton);

        sprrzRadioButton = new QRadioButton(UserCodeButtonGroup);
        sprrzRadioButton->setObjectName(QString::fromUtf8("sprrzRadioButton"));
        sprrzRadioButton->setFocusPolicy(Qt::NoFocus);

        vboxLayout->addWidget(sprrzRadioButton);

        flurzRadioButton = new QRadioButton(UserCodeButtonGroup);
        flurzRadioButton->setObjectName(QString::fromUtf8("flurzRadioButton"));

        vboxLayout->addWidget(flurzRadioButton);


        hboxLayout->addWidget(UserCodeButtonGroup);

        TargetbuttonGroup = new QGroupBox(GItab);
        TargetbuttonGroup->setObjectName(QString::fromUtf8("TargetbuttonGroup"));
        vboxLayout1 = new QVBoxLayout(TargetbuttonGroup);
        vboxLayout1->setSpacing(6);
        vboxLayout1->setContentsMargins(11, 11, 11, 11);
        vboxLayout1->setObjectName(QString::fromUtf8("vboxLayout1"));
        vboxLayout2 = new QVBoxLayout();
        vboxLayout2->setSpacing(6);
        vboxLayout2->setObjectName(QString::fromUtf8("vboxLayout2"));
        OptradioButton = new QRadioButton(TargetbuttonGroup);
        OptradioButton->setObjectName(QString::fromUtf8("OptradioButton"));
        OptradioButton->setChecked(true);

        vboxLayout2->addWidget(OptradioButton);

        NoOptradioButton = new QRadioButton(TargetbuttonGroup);
        NoOptradioButton->setObjectName(QString::fromUtf8("NoOptradioButton"));

        vboxLayout2->addWidget(NoOptradioButton);

        DebugradioButton = new QRadioButton(TargetbuttonGroup);
        DebugradioButton->setObjectName(QString::fromUtf8("DebugradioButton"));

        vboxLayout2->addWidget(DebugradioButton);

        CleanradioButton = new QRadioButton(TargetbuttonGroup);
        CleanradioButton->setObjectName(QString::fromUtf8("CleanradioButton"));

        vboxLayout2->addWidget(CleanradioButton);


        vboxLayout1->addLayout(vboxLayout2);


        hboxLayout->addWidget(TargetbuttonGroup);

        UserCodeAreaButtonGroup = new QGroupBox(GItab);
        UserCodeAreaButtonGroup->setObjectName(QString::fromUtf8("UserCodeAreaButtonGroup"));
        vboxLayout3 = new QVBoxLayout(UserCodeAreaButtonGroup);
        vboxLayout3->setSpacing(6);
        vboxLayout3->setContentsMargins(11, 11, 11, 11);
        vboxLayout3->setObjectName(QString::fromUtf8("vboxLayout3"));
        vboxLayout4 = new QVBoxLayout();
        vboxLayout4->setSpacing(6);
        vboxLayout4->setObjectName(QString::fromUtf8("vboxLayout4"));
        HOMERadioButton = new QRadioButton(UserCodeAreaButtonGroup);
        HOMERadioButton->setObjectName(QString::fromUtf8("HOMERadioButton"));
        HOMERadioButton->setChecked(true);

        vboxLayout4->addWidget(HOMERadioButton);

        HEN_HOUSERadioButton = new QRadioButton(UserCodeAreaButtonGroup);
        HEN_HOUSERadioButton->setObjectName(QString::fromUtf8("HEN_HOUSERadioButton"));

        vboxLayout4->addWidget(HEN_HOUSERadioButton);

        OtherAreaRadioButton = new QRadioButton(UserCodeAreaButtonGroup);
        OtherAreaRadioButton->setObjectName(QString::fromUtf8("OtherAreaRadioButton"));

        vboxLayout4->addWidget(OtherAreaRadioButton);


        vboxLayout3->addLayout(vboxLayout4);


        hboxLayout->addWidget(UserCodeAreaButtonGroup);

        PEGSDataAreaButtonGroup = new QGroupBox(GItab);
        PEGSDataAreaButtonGroup->setObjectName(QString::fromUtf8("PEGSDataAreaButtonGroup"));
        vboxLayout5 = new QVBoxLayout(PEGSDataAreaButtonGroup);
        vboxLayout5->setSpacing(6);
        vboxLayout5->setContentsMargins(11, 11, 11, 11);
        vboxLayout5->setObjectName(QString::fromUtf8("vboxLayout5"));
        vboxLayout6 = new QVBoxLayout();
        vboxLayout6->setSpacing(6);
        vboxLayout6->setObjectName(QString::fromUtf8("vboxLayout6"));
        HOMEPegsRadioButton = new QRadioButton(PEGSDataAreaButtonGroup);
        HOMEPegsRadioButton->setObjectName(QString::fromUtf8("HOMEPegsRadioButton"));

        vboxLayout6->addWidget(HOMEPegsRadioButton);

        HEN_HOUSEPegsRadioButton = new QRadioButton(PEGSDataAreaButtonGroup);
        HEN_HOUSEPegsRadioButton->setObjectName(QString::fromUtf8("HEN_HOUSEPegsRadioButton"));
        HEN_HOUSEPegsRadioButton->setChecked(true);

        vboxLayout6->addWidget(HEN_HOUSEPegsRadioButton);

        OtherPegsAreaRadioButton = new QRadioButton(PEGSDataAreaButtonGroup);
        OtherPegsAreaRadioButton->setObjectName(QString::fromUtf8("OtherPegsAreaRadioButton"));

        vboxLayout6->addWidget(OtherPegsAreaRadioButton);

        PEGSlessRadioButton = new QRadioButton(PEGSDataAreaButtonGroup);
        PEGSlessRadioButton->setObjectName(QString::fromUtf8("PEGSlessRadioButton"));

        vboxLayout6->addWidget(PEGSlessRadioButton);


        vboxLayout5->addLayout(vboxLayout6);


        hboxLayout->addWidget(PEGSDataAreaButtonGroup);


        gridLayout->addLayout(hboxLayout, 1, 0, 1, 1);

        TitleGroupBox = new QGroupBox(GItab);
        TitleGroupBox->setObjectName(QString::fromUtf8("TitleGroupBox"));
        vboxLayout7 = new QVBoxLayout(TitleGroupBox);
        vboxLayout7->setSpacing(6);
        vboxLayout7->setContentsMargins(11, 11, 11, 11);
        vboxLayout7->setObjectName(QString::fromUtf8("vboxLayout7"));
        TitleEdit = new QLineEdit(TitleGroupBox);
        TitleEdit->setObjectName(QString::fromUtf8("TitleEdit"));
        QSizePolicy sizePolicy4(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy4.setHorizontalStretch(0);
        sizePolicy4.setVerticalStretch(0);
        sizePolicy4.setHeightForWidth(TitleEdit->sizePolicy().hasHeightForWidth());
        TitleEdit->setSizePolicy(sizePolicy4);
        TitleEdit->setMaxLength(80);

        vboxLayout7->addWidget(TitleEdit);


        gridLayout->addWidget(TitleGroupBox, 0, 0, 1, 1);

        theguyLabel = new QLabel(GItab);
        theguyLabel->setObjectName(QString::fromUtf8("theguyLabel"));
        QSizePolicy sizePolicy5(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy5.setHorizontalStretch(0);
        sizePolicy5.setVerticalStretch(0);
        sizePolicy5.setHeightForWidth(theguyLabel->sizePolicy().hasHeightForWidth());
        theguyLabel->setSizePolicy(sizePolicy5);
        theguyLabel->setMinimumSize(QSize(153, 200));
        theguyLabel->setMaximumSize(QSize(181, 185));
        theguyLabel->setFrameShape(QFrame::Box);
        theguyLabel->setLineWidth(0);
        theguyLabel->setScaledContents(true);
        theguyLabel->setWordWrap(false);

        gridLayout->addWidget(theguyLabel, 0, 1, 2, 1);


        verticalLayout_28->addLayout(gridLayout);

        verticalLayout_26 = new QVBoxLayout();
        verticalLayout_26->setSpacing(6);
        verticalLayout_26->setObjectName(QString::fromUtf8("verticalLayout_26"));
        InputFileGroupBox = new QGroupBox(GItab);
        InputFileGroupBox->setObjectName(QString::fromUtf8("InputFileGroupBox"));
        horizontalLayout_21 = new QHBoxLayout(InputFileGroupBox);
        horizontalLayout_21->setSpacing(6);
        horizontalLayout_21->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_21->setObjectName(QString::fromUtf8("horizontalLayout_21"));
        horizontalLayout_18 = new QHBoxLayout();
        horizontalLayout_18->setSpacing(6);
        horizontalLayout_18->setObjectName(QString::fromUtf8("horizontalLayout_18"));
        InputFileComboBox = new QComboBox(InputFileGroupBox);
        InputFileComboBox->setObjectName(QString::fromUtf8("InputFileComboBox"));
        QSizePolicy sizePolicy6(QSizePolicy::Minimum, QSizePolicy::Fixed);
        sizePolicy6.setHorizontalStretch(232);
        sizePolicy6.setVerticalStretch(0);
        sizePolicy6.setHeightForWidth(InputFileComboBox->sizePolicy().hasHeightForWidth());
        InputFileComboBox->setSizePolicy(sizePolicy6);
        InputFileComboBox->setMinimumSize(QSize(0, 28));
        InputFileComboBox->setEditable(true);
        InputFileComboBox->setInsertPolicy(QComboBox::InsertAtTop);
        InputFileComboBox->setAutoCompletion(true);
        InputFileComboBox->setDuplicatesEnabled(false);

        horizontalLayout_18->addWidget(InputFileComboBox);

        OpenFileButton = new QPushButton(InputFileGroupBox);
        OpenFileButton->setObjectName(QString::fromUtf8("OpenFileButton"));
        QSizePolicy sizePolicy7(QSizePolicy::Minimum, QSizePolicy::Fixed);
        sizePolicy7.setHorizontalStretch(1);
        sizePolicy7.setVerticalStretch(0);
        sizePolicy7.setHeightForWidth(OpenFileButton->sizePolicy().hasHeightForWidth());
        OpenFileButton->setSizePolicy(sizePolicy7);
        OpenFileButton->setMinimumSize(QSize(0, 29));
        OpenFileButton->setAutoDefault(false);

        horizontalLayout_18->addWidget(OpenFileButton);


        horizontalLayout_21->addLayout(horizontalLayout_18);


        verticalLayout_26->addWidget(InputFileGroupBox);

        pegs4GroupBox = new QGroupBox(GItab);
        pegs4GroupBox->setObjectName(QString::fromUtf8("pegs4GroupBox"));
        hboxLayout1 = new QHBoxLayout(pegs4GroupBox);
        hboxLayout1->setSpacing(6);
        hboxLayout1->setContentsMargins(11, 11, 11, 11);
        hboxLayout1->setObjectName(QString::fromUtf8("hboxLayout1"));
        hboxLayout2 = new QHBoxLayout();
        hboxLayout2->setSpacing(6);
        hboxLayout2->setObjectName(QString::fromUtf8("hboxLayout2"));
        pegs4ComboBox = new QComboBox(pegs4GroupBox);
        pegs4ComboBox->setObjectName(QString::fromUtf8("pegs4ComboBox"));
        sizePolicy6.setHeightForWidth(pegs4ComboBox->sizePolicy().hasHeightForWidth());
        pegs4ComboBox->setSizePolicy(sizePolicy6);
        pegs4ComboBox->setMinimumSize(QSize(0, 28));
        pegs4ComboBox->setEditable(true);
        pegs4ComboBox->setInsertPolicy(QComboBox::InsertAtTop);
        pegs4ComboBox->setAutoCompletion(true);

        hboxLayout2->addWidget(pegs4ComboBox);

        Pegs4FileButton = new QPushButton(pegs4GroupBox);
        Pegs4FileButton->setObjectName(QString::fromUtf8("Pegs4FileButton"));
        sizePolicy7.setHeightForWidth(Pegs4FileButton->sizePolicy().hasHeightForWidth());
        Pegs4FileButton->setSizePolicy(sizePolicy7);
        Pegs4FileButton->setMinimumSize(QSize(0, 29));
        Pegs4FileButton->setAutoDefault(false);

        hboxLayout2->addWidget(Pegs4FileButton);


        hboxLayout1->addLayout(hboxLayout2);


        verticalLayout_26->addWidget(pegs4GroupBox);

        groupBox57 = new QGroupBox(GItab);
        groupBox57->setObjectName(QString::fromUtf8("groupBox57"));
        hboxLayout3 = new QHBoxLayout(groupBox57);
        hboxLayout3->setSpacing(6);
        hboxLayout3->setContentsMargins(11, 11, 11, 11);
        hboxLayout3->setObjectName(QString::fromUtf8("hboxLayout3"));
        hboxLayout4 = new QHBoxLayout();
        hboxLayout4->setSpacing(6);
        hboxLayout4->setObjectName(QString::fromUtf8("hboxLayout4"));
        CONFcomboBox = new QComboBox(groupBox57);
        CONFcomboBox->setObjectName(QString::fromUtf8("CONFcomboBox"));
        sizePolicy6.setHeightForWidth(CONFcomboBox->sizePolicy().hasHeightForWidth());
        CONFcomboBox->setSizePolicy(sizePolicy6);
        CONFcomboBox->setMinimumSize(QSize(0, 28));
        CONFcomboBox->setEditable(true);
        CONFcomboBox->setInsertPolicy(QComboBox::InsertAtTop);
        CONFcomboBox->setAutoCompletion(true);
        CONFcomboBox->setDuplicatesEnabled(false);

        hboxLayout4->addWidget(CONFcomboBox);

        SPECButton = new QPushButton(groupBox57);
        SPECButton->setObjectName(QString::fromUtf8("SPECButton"));
        sizePolicy7.setHeightForWidth(SPECButton->sizePolicy().hasHeightForWidth());
        SPECButton->setSizePolicy(sizePolicy7);
        SPECButton->setMinimumSize(QSize(0, 29));

        hboxLayout4->addWidget(SPECButton);


        hboxLayout3->addLayout(hboxLayout4);


        verticalLayout_26->addWidget(groupBox57);

        hboxLayout5 = new QHBoxLayout();
        hboxLayout5->setSpacing(6);
        hboxLayout5->setObjectName(QString::fromUtf8("hboxLayout5"));
        ConfigurationButton = new QPushButton(GItab);
        ConfigurationButton->setObjectName(QString::fromUtf8("ConfigurationButton"));

        hboxLayout5->addWidget(ConfigurationButton);

        spacer45 = new QSpacerItem(521, 21, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout5->addItem(spacer45);

        logfileButton = new QPushButton(GItab);
        logfileButton->setObjectName(QString::fromUtf8("logfileButton"));
        logfileButton->setEnabled(false);
        logfileButton->setAutoDefault(false);

        hboxLayout5->addWidget(logfileButton);


        verticalLayout_26->addLayout(hboxLayout5);


        verticalLayout_28->addLayout(verticalLayout_26);


        verticalLayout_30->addLayout(verticalLayout_28);

        TabWidgetRZ->addTab(GItab, QString());
        IOtab = new QWidget();
        IOtab->setObjectName(QString::fromUtf8("IOtab"));
        hboxLayout6 = new QHBoxLayout(IOtab);
        hboxLayout6->setSpacing(6);
        hboxLayout6->setContentsMargins(11, 11, 11, 11);
        hboxLayout6->setObjectName(QString::fromUtf8("hboxLayout6"));
        hboxLayout7 = new QHBoxLayout();
        hboxLayout7->setSpacing(6);
        hboxLayout7->setObjectName(QString::fromUtf8("hboxLayout7"));
        vboxLayout8 = new QVBoxLayout();
        vboxLayout8->setSpacing(6);
        vboxLayout8->setObjectName(QString::fromUtf8("vboxLayout8"));
        randNButtonGroup = new QGroupBox(IOtab);
        randNButtonGroup->setObjectName(QString::fromUtf8("randNButtonGroup"));
        vboxLayout9 = new QVBoxLayout(randNButtonGroup);
        vboxLayout9->setSpacing(6);
        vboxLayout9->setContentsMargins(11, 11, 11, 11);
        vboxLayout9->setObjectName(QString::fromUtf8("vboxLayout9"));
        vboxLayout10 = new QVBoxLayout();
        vboxLayout10->setSpacing(6);
        vboxLayout10->setObjectName(QString::fromUtf8("vboxLayout10"));
        noRandRadioButton = new QRadioButton(randNButtonGroup);
        noRandRadioButton->setObjectName(QString::fromUtf8("noRandRadioButton"));
        noRandRadioButton->setChecked(true);

        vboxLayout10->addWidget(noRandRadioButton);

        lastRandRadioButton = new QRadioButton(randNButtonGroup);
        lastRandRadioButton->setObjectName(QString::fromUtf8("lastRandRadioButton"));

        vboxLayout10->addWidget(lastRandRadioButton);

        allRandRadioButton = new QRadioButton(randNButtonGroup);
        allRandRadioButton->setObjectName(QString::fromUtf8("allRandRadioButton"));

        vboxLayout10->addWidget(allRandRadioButton);


        vboxLayout9->addLayout(vboxLayout10);


        vboxLayout8->addWidget(randNButtonGroup);

        DoseRegGroupBox = new QGroupBox(IOtab);
        DoseRegGroupBox->setObjectName(QString::fromUtf8("DoseRegGroupBox"));
        DoseRegGroupBox->setEnabled(true);
        vboxLayout11 = new QVBoxLayout(DoseRegGroupBox);
        vboxLayout11->setSpacing(6);
        vboxLayout11->setContentsMargins(11, 11, 11, 11);
        vboxLayout11->setObjectName(QString::fromUtf8("vboxLayout11"));
        vboxLayout12 = new QVBoxLayout();
        vboxLayout12->setSpacing(6);
        vboxLayout12->setObjectName(QString::fromUtf8("vboxLayout12"));
        hboxLayout8 = new QHBoxLayout();
        hboxLayout8->setSpacing(6);
        hboxLayout8->setObjectName(QString::fromUtf8("hboxLayout8"));
        minPlnTextLabel = new QLabel(DoseRegGroupBox);
        minPlnTextLabel->setObjectName(QString::fromUtf8("minPlnTextLabel"));
        minPlnTextLabel->setWordWrap(false);

        hboxLayout8->addWidget(minPlnTextLabel);

        Spacer1 = new QSpacerItem(20, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout8->addItem(Spacer1);

        minPlnSpinBox = new QSpinBox(DoseRegGroupBox);
        minPlnSpinBox->setObjectName(QString::fromUtf8("minPlnSpinBox"));
        minPlnSpinBox->setMinimum(1);
        minPlnSpinBox->setMaximum(1001);
        minPlnSpinBox->setValue(1);

        hboxLayout8->addWidget(minPlnSpinBox);


        vboxLayout12->addLayout(hboxLayout8);

        hboxLayout9 = new QHBoxLayout();
        hboxLayout9->setSpacing(6);
        hboxLayout9->setObjectName(QString::fromUtf8("hboxLayout9"));
        maxPlnTextLabel = new QLabel(DoseRegGroupBox);
        maxPlnTextLabel->setObjectName(QString::fromUtf8("maxPlnTextLabel"));
        maxPlnTextLabel->setWordWrap(false);

        hboxLayout9->addWidget(maxPlnTextLabel);

        Spacer2 = new QSpacerItem(20, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout9->addItem(Spacer2);

        maxPlnSpinBox = new QSpinBox(DoseRegGroupBox);
        maxPlnSpinBox->setObjectName(QString::fromUtf8("maxPlnSpinBox"));
        maxPlnSpinBox->setMinimum(1);
        maxPlnSpinBox->setMaximum(1001);
        maxPlnSpinBox->setValue(61);

        hboxLayout9->addWidget(maxPlnSpinBox);


        vboxLayout12->addLayout(hboxLayout9);

        hboxLayout10 = new QHBoxLayout();
        hboxLayout10->setSpacing(6);
        hboxLayout10->setObjectName(QString::fromUtf8("hboxLayout10"));
        minCylTextLabel = new QLabel(DoseRegGroupBox);
        minCylTextLabel->setObjectName(QString::fromUtf8("minCylTextLabel"));
        minCylTextLabel->setWordWrap(false);

        hboxLayout10->addWidget(minCylTextLabel);

        Spacer3 = new QSpacerItem(20, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout10->addItem(Spacer3);

        minCylSpinBox = new QSpinBox(DoseRegGroupBox);
        minCylSpinBox->setObjectName(QString::fromUtf8("minCylSpinBox"));
        minCylSpinBox->setMaximum(1000);

        hboxLayout10->addWidget(minCylSpinBox);


        vboxLayout12->addLayout(hboxLayout10);

        hboxLayout11 = new QHBoxLayout();
        hboxLayout11->setSpacing(6);
        hboxLayout11->setObjectName(QString::fromUtf8("hboxLayout11"));
        maxCylTextLabel = new QLabel(DoseRegGroupBox);
        maxCylTextLabel->setObjectName(QString::fromUtf8("maxCylTextLabel"));
        maxCylTextLabel->setWordWrap(false);

        hboxLayout11->addWidget(maxCylTextLabel);

        Spacer4 = new QSpacerItem(20, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout11->addItem(Spacer4);

        maxCylSpinBox = new QSpinBox(DoseRegGroupBox);
        maxCylSpinBox->setObjectName(QString::fromUtf8("maxCylSpinBox"));
        maxCylSpinBox->setMaximum(1000);
        maxCylSpinBox->setValue(60);

        hboxLayout11->addWidget(maxCylSpinBox);


        vboxLayout12->addLayout(hboxLayout11);


        vboxLayout11->addLayout(vboxLayout12);


        vboxLayout8->addWidget(DoseRegGroupBox);

        Frame4 = new QFrame(IOtab);
        Frame4->setObjectName(QString::fromUtf8("Frame4"));
        Frame4->setFrameShape(QFrame::StyledPanel);
        Frame4->setFrameShadow(QFrame::Sunken);
        vboxLayout13 = new QVBoxLayout(Frame4);
        vboxLayout13->setSpacing(6);
        vboxLayout13->setContentsMargins(11, 11, 11, 11);
        vboxLayout13->setObjectName(QString::fromUtf8("vboxLayout13"));
        storeDataCheckBox = new QCheckBox(Frame4);
        storeDataCheckBox->setObjectName(QString::fromUtf8("storeDataCheckBox"));
        storeDataCheckBox->setChecked(true);

        vboxLayout13->addWidget(storeDataCheckBox);


        vboxLayout8->addWidget(Frame4);

        spacer41 = new QSpacerItem(20, 90, QSizePolicy::Minimum, QSizePolicy::Expanding);

        vboxLayout8->addItem(spacer41);


        hboxLayout7->addLayout(vboxLayout8);

        vboxLayout14 = new QVBoxLayout();
        vboxLayout14->setSpacing(6);
        vboxLayout14->setObjectName(QString::fromUtf8("vboxLayout14"));
        Frame6 = new QFrame(IOtab);
        Frame6->setObjectName(QString::fromUtf8("Frame6"));
        Frame6->setFrameShape(QFrame::StyledPanel);
        Frame6->setFrameShadow(QFrame::Sunken);
        vboxLayout15 = new QVBoxLayout(Frame6);
        vboxLayout15->setSpacing(6);
        vboxLayout15->setContentsMargins(11, 11, 11, 11);
        vboxLayout15->setObjectName(QString::fromUtf8("vboxLayout15"));
        vboxLayout16 = new QVBoxLayout();
        vboxLayout16->setSpacing(6);
        vboxLayout16->setObjectName(QString::fromUtf8("vboxLayout16"));
        hboxLayout12 = new QHBoxLayout();
        hboxLayout12->setSpacing(6);
        hboxLayout12->setObjectName(QString::fromUtf8("hboxLayout12"));
        outoptTextLabel = new QLabel(Frame6);
        outoptTextLabel->setObjectName(QString::fromUtf8("outoptTextLabel"));
        outoptTextLabel->setWordWrap(false);

        hboxLayout12->addWidget(outoptTextLabel);

        Spacer78 = new QSpacerItem(21, 20, QSizePolicy::Fixed, QSizePolicy::Minimum);

        hboxLayout12->addItem(Spacer78);

        outoptComboBox = new QComboBox(Frame6);
        outoptComboBox->addItem(QString());
        outoptComboBox->addItem(QString());
        outoptComboBox->addItem(QString());
        outoptComboBox->setObjectName(QString::fromUtf8("outoptComboBox"));
        sizePolicy4.setHeightForWidth(outoptComboBox->sizePolicy().hasHeightForWidth());
        outoptComboBox->setSizePolicy(sizePolicy4);
        outoptComboBox->setMaxCount(6);

        hboxLayout12->addWidget(outoptComboBox);


        vboxLayout16->addLayout(hboxLayout12);

        hboxLayout13 = new QHBoxLayout();
        hboxLayout13->setSpacing(6);
        hboxLayout13->setObjectName(QString::fromUtf8("hboxLayout13"));
        etransportLabel = new QLabel(Frame6);
        etransportLabel->setObjectName(QString::fromUtf8("etransportLabel"));
        etransportLabel->setEnabled(false);
        etransportLabel->setWordWrap(false);

        hboxLayout13->addWidget(etransportLabel);

        Spacer5 = new QSpacerItem(20, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout13->addItem(Spacer5);

        etransportComboBox = new QComboBox(Frame6);
        etransportComboBox->addItem(QString());
        etransportComboBox->addItem(QString());
        etransportComboBox->setObjectName(QString::fromUtf8("etransportComboBox"));
        etransportComboBox->setEnabled(false);
        etransportComboBox->setMaxCount(6);

        hboxLayout13->addWidget(etransportComboBox);


        vboxLayout16->addLayout(hboxLayout13);

        hboxLayout14 = new QHBoxLayout();
        hboxLayout14->setSpacing(6);
        hboxLayout14->setObjectName(QString::fromUtf8("hboxLayout14"));
        iwatchTextLabel = new QLabel(Frame6);
        iwatchTextLabel->setObjectName(QString::fromUtf8("iwatchTextLabel"));
        iwatchTextLabel->setWordWrap(false);

        hboxLayout14->addWidget(iwatchTextLabel);

        Spacer8 = new QSpacerItem(20, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout14->addItem(Spacer8);

        iwatchComboBox = new QComboBox(Frame6);
        iwatchComboBox->addItem(QString());
        iwatchComboBox->addItem(QString());
        iwatchComboBox->addItem(QString());
        iwatchComboBox->addItem(QString());
        iwatchComboBox->addItem(QString());
        iwatchComboBox->setObjectName(QString::fromUtf8("iwatchComboBox"));

        hboxLayout14->addWidget(iwatchComboBox);


        vboxLayout16->addLayout(hboxLayout14);

        hboxLayout15 = new QHBoxLayout();
        hboxLayout15->setSpacing(6);
        hboxLayout15->setObjectName(QString::fromUtf8("hboxLayout15"));
        restartTextLabel = new QLabel(Frame6);
        restartTextLabel->setObjectName(QString::fromUtf8("restartTextLabel"));
        restartTextLabel->setWordWrap(false);

        hboxLayout15->addWidget(restartTextLabel);

        Spacer7 = new QSpacerItem(20, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout15->addItem(Spacer7);

        irestartComboBox = new QComboBox(Frame6);
        irestartComboBox->addItem(QString());
        irestartComboBox->addItem(QString());
        irestartComboBox->addItem(QString());
        irestartComboBox->addItem(QString());
        irestartComboBox->addItem(QString());
        irestartComboBox->setObjectName(QString::fromUtf8("irestartComboBox"));
        irestartComboBox->setMaxCount(6);

        hboxLayout15->addWidget(irestartComboBox);


        vboxLayout16->addLayout(hboxLayout15);


        vboxLayout15->addLayout(vboxLayout16);


        vboxLayout14->addWidget(Frame6);

        SPRRegGroupBox = new QGroupBox(IOtab);
        SPRRegGroupBox->setObjectName(QString::fromUtf8("SPRRegGroupBox"));
        SPRRegGroupBox->setEnabled(true);
        sproutComboBox = new QComboBox(SPRRegGroupBox);
        sproutComboBox->addItem(QString());
        sproutComboBox->addItem(QString());
        sproutComboBox->setObjectName(QString::fromUtf8("sproutComboBox"));
        sproutComboBox->setGeometry(QRect(192, 25, 120, 22));
        sproutTextLabel = new QLabel(SPRRegGroupBox);
        sproutTextLabel->setObjectName(QString::fromUtf8("sproutTextLabel"));
        sproutTextLabel->setGeometry(QRect(12, 25, 170, 25));
        sproutTextLabel->setWordWrap(false);
        sproutTable = new QTableWidget(SPRRegGroupBox);
        if (sproutTable->columnCount() < 2)
            sproutTable->setColumnCount(2);
        QTableWidgetItem *__qtablewidgetitem = new QTableWidgetItem();
        sproutTable->setHorizontalHeaderItem(0, __qtablewidgetitem);
        QTableWidgetItem *__qtablewidgetitem1 = new QTableWidgetItem();
        sproutTable->setHorizontalHeaderItem(1, __qtablewidgetitem1);
        if (sproutTable->rowCount() < 50)
            sproutTable->setRowCount(50);
        QTableWidgetItem *__qtablewidgetitem2 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(0, __qtablewidgetitem2);
        QTableWidgetItem *__qtablewidgetitem3 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(1, __qtablewidgetitem3);
        QTableWidgetItem *__qtablewidgetitem4 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(2, __qtablewidgetitem4);
        QTableWidgetItem *__qtablewidgetitem5 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(3, __qtablewidgetitem5);
        QTableWidgetItem *__qtablewidgetitem6 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(4, __qtablewidgetitem6);
        QTableWidgetItem *__qtablewidgetitem7 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(5, __qtablewidgetitem7);
        QTableWidgetItem *__qtablewidgetitem8 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(6, __qtablewidgetitem8);
        QTableWidgetItem *__qtablewidgetitem9 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(7, __qtablewidgetitem9);
        QTableWidgetItem *__qtablewidgetitem10 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(8, __qtablewidgetitem10);
        QTableWidgetItem *__qtablewidgetitem11 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(9, __qtablewidgetitem11);
        QTableWidgetItem *__qtablewidgetitem12 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(10, __qtablewidgetitem12);
        QTableWidgetItem *__qtablewidgetitem13 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(11, __qtablewidgetitem13);
        QTableWidgetItem *__qtablewidgetitem14 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(12, __qtablewidgetitem14);
        QTableWidgetItem *__qtablewidgetitem15 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(13, __qtablewidgetitem15);
        QTableWidgetItem *__qtablewidgetitem16 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(14, __qtablewidgetitem16);
        QTableWidgetItem *__qtablewidgetitem17 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(15, __qtablewidgetitem17);
        QTableWidgetItem *__qtablewidgetitem18 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(16, __qtablewidgetitem18);
        QTableWidgetItem *__qtablewidgetitem19 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(17, __qtablewidgetitem19);
        QTableWidgetItem *__qtablewidgetitem20 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(18, __qtablewidgetitem20);
        QTableWidgetItem *__qtablewidgetitem21 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(19, __qtablewidgetitem21);
        QTableWidgetItem *__qtablewidgetitem22 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(20, __qtablewidgetitem22);
        QTableWidgetItem *__qtablewidgetitem23 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(21, __qtablewidgetitem23);
        QTableWidgetItem *__qtablewidgetitem24 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(22, __qtablewidgetitem24);
        QTableWidgetItem *__qtablewidgetitem25 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(23, __qtablewidgetitem25);
        QTableWidgetItem *__qtablewidgetitem26 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(24, __qtablewidgetitem26);
        QTableWidgetItem *__qtablewidgetitem27 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(25, __qtablewidgetitem27);
        QTableWidgetItem *__qtablewidgetitem28 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(26, __qtablewidgetitem28);
        QTableWidgetItem *__qtablewidgetitem29 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(27, __qtablewidgetitem29);
        QTableWidgetItem *__qtablewidgetitem30 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(28, __qtablewidgetitem30);
        QTableWidgetItem *__qtablewidgetitem31 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(29, __qtablewidgetitem31);
        QTableWidgetItem *__qtablewidgetitem32 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(30, __qtablewidgetitem32);
        QTableWidgetItem *__qtablewidgetitem33 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(31, __qtablewidgetitem33);
        QTableWidgetItem *__qtablewidgetitem34 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(32, __qtablewidgetitem34);
        QTableWidgetItem *__qtablewidgetitem35 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(33, __qtablewidgetitem35);
        QTableWidgetItem *__qtablewidgetitem36 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(34, __qtablewidgetitem36);
        QTableWidgetItem *__qtablewidgetitem37 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(35, __qtablewidgetitem37);
        QTableWidgetItem *__qtablewidgetitem38 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(36, __qtablewidgetitem38);
        QTableWidgetItem *__qtablewidgetitem39 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(37, __qtablewidgetitem39);
        QTableWidgetItem *__qtablewidgetitem40 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(38, __qtablewidgetitem40);
        QTableWidgetItem *__qtablewidgetitem41 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(39, __qtablewidgetitem41);
        QTableWidgetItem *__qtablewidgetitem42 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(40, __qtablewidgetitem42);
        QTableWidgetItem *__qtablewidgetitem43 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(41, __qtablewidgetitem43);
        QTableWidgetItem *__qtablewidgetitem44 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(42, __qtablewidgetitem44);
        QTableWidgetItem *__qtablewidgetitem45 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(43, __qtablewidgetitem45);
        QTableWidgetItem *__qtablewidgetitem46 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(44, __qtablewidgetitem46);
        QTableWidgetItem *__qtablewidgetitem47 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(45, __qtablewidgetitem47);
        QTableWidgetItem *__qtablewidgetitem48 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(46, __qtablewidgetitem48);
        QTableWidgetItem *__qtablewidgetitem49 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(47, __qtablewidgetitem49);
        QTableWidgetItem *__qtablewidgetitem50 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(48, __qtablewidgetitem50);
        QTableWidgetItem *__qtablewidgetitem51 = new QTableWidgetItem();
        sproutTable->setVerticalHeaderItem(49, __qtablewidgetitem51);
        sproutTable->setObjectName(QString::fromUtf8("sproutTable"));
        sproutTable->setGeometry(QRect(9, 54, 251, 170));
        sproutTable->setRowCount(50);
        sproutTable->setProperty("ColumnCount", QVariant(2));

        vboxLayout14->addWidget(SPRRegGroupBox);


        hboxLayout7->addLayout(vboxLayout14);

        vboxLayout17 = new QVBoxLayout();
        vboxLayout17->setSpacing(6);
        vboxLayout17->setObjectName(QString::fromUtf8("vboxLayout17"));
        PrintFluSpecGroupBox = new QGroupBox(IOtab);
        PrintFluSpecGroupBox->setObjectName(QString::fromUtf8("PrintFluSpecGroupBox"));
        PrintFluSpeComboBox = new QComboBox(PrintFluSpecGroupBox);
        PrintFluSpeComboBox->addItem(QString());
        PrintFluSpeComboBox->addItem(QString());
        PrintFluSpeComboBox->addItem(QString());
        PrintFluSpeComboBox->addItem(QString());
        PrintFluSpeComboBox->setObjectName(QString::fromUtf8("PrintFluSpeComboBox"));
        PrintFluSpeComboBox->setGeometry(QRect(12, 21, 160, 22));
        ListFluTable = new QTableWidget(PrintFluSpecGroupBox);
        if (ListFluTable->columnCount() < 2)
            ListFluTable->setColumnCount(2);
        QTableWidgetItem *__qtablewidgetitem52 = new QTableWidgetItem();
        ListFluTable->setHorizontalHeaderItem(0, __qtablewidgetitem52);
        QTableWidgetItem *__qtablewidgetitem53 = new QTableWidgetItem();
        ListFluTable->setHorizontalHeaderItem(1, __qtablewidgetitem53);
        if (ListFluTable->rowCount() < 50)
            ListFluTable->setRowCount(50);
        QTableWidgetItem *__qtablewidgetitem54 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(0, __qtablewidgetitem54);
        QTableWidgetItem *__qtablewidgetitem55 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(1, __qtablewidgetitem55);
        QTableWidgetItem *__qtablewidgetitem56 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(2, __qtablewidgetitem56);
        QTableWidgetItem *__qtablewidgetitem57 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(3, __qtablewidgetitem57);
        QTableWidgetItem *__qtablewidgetitem58 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(4, __qtablewidgetitem58);
        QTableWidgetItem *__qtablewidgetitem59 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(5, __qtablewidgetitem59);
        QTableWidgetItem *__qtablewidgetitem60 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(6, __qtablewidgetitem60);
        QTableWidgetItem *__qtablewidgetitem61 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(7, __qtablewidgetitem61);
        QTableWidgetItem *__qtablewidgetitem62 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(8, __qtablewidgetitem62);
        QTableWidgetItem *__qtablewidgetitem63 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(9, __qtablewidgetitem63);
        QTableWidgetItem *__qtablewidgetitem64 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(10, __qtablewidgetitem64);
        QTableWidgetItem *__qtablewidgetitem65 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(11, __qtablewidgetitem65);
        QTableWidgetItem *__qtablewidgetitem66 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(12, __qtablewidgetitem66);
        QTableWidgetItem *__qtablewidgetitem67 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(13, __qtablewidgetitem67);
        QTableWidgetItem *__qtablewidgetitem68 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(14, __qtablewidgetitem68);
        QTableWidgetItem *__qtablewidgetitem69 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(15, __qtablewidgetitem69);
        QTableWidgetItem *__qtablewidgetitem70 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(16, __qtablewidgetitem70);
        QTableWidgetItem *__qtablewidgetitem71 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(17, __qtablewidgetitem71);
        QTableWidgetItem *__qtablewidgetitem72 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(18, __qtablewidgetitem72);
        QTableWidgetItem *__qtablewidgetitem73 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(19, __qtablewidgetitem73);
        QTableWidgetItem *__qtablewidgetitem74 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(20, __qtablewidgetitem74);
        QTableWidgetItem *__qtablewidgetitem75 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(21, __qtablewidgetitem75);
        QTableWidgetItem *__qtablewidgetitem76 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(22, __qtablewidgetitem76);
        QTableWidgetItem *__qtablewidgetitem77 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(23, __qtablewidgetitem77);
        QTableWidgetItem *__qtablewidgetitem78 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(24, __qtablewidgetitem78);
        QTableWidgetItem *__qtablewidgetitem79 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(25, __qtablewidgetitem79);
        QTableWidgetItem *__qtablewidgetitem80 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(26, __qtablewidgetitem80);
        QTableWidgetItem *__qtablewidgetitem81 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(27, __qtablewidgetitem81);
        QTableWidgetItem *__qtablewidgetitem82 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(28, __qtablewidgetitem82);
        QTableWidgetItem *__qtablewidgetitem83 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(29, __qtablewidgetitem83);
        QTableWidgetItem *__qtablewidgetitem84 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(30, __qtablewidgetitem84);
        QTableWidgetItem *__qtablewidgetitem85 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(31, __qtablewidgetitem85);
        QTableWidgetItem *__qtablewidgetitem86 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(32, __qtablewidgetitem86);
        QTableWidgetItem *__qtablewidgetitem87 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(33, __qtablewidgetitem87);
        QTableWidgetItem *__qtablewidgetitem88 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(34, __qtablewidgetitem88);
        QTableWidgetItem *__qtablewidgetitem89 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(35, __qtablewidgetitem89);
        QTableWidgetItem *__qtablewidgetitem90 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(36, __qtablewidgetitem90);
        QTableWidgetItem *__qtablewidgetitem91 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(37, __qtablewidgetitem91);
        QTableWidgetItem *__qtablewidgetitem92 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(38, __qtablewidgetitem92);
        QTableWidgetItem *__qtablewidgetitem93 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(39, __qtablewidgetitem93);
        QTableWidgetItem *__qtablewidgetitem94 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(40, __qtablewidgetitem94);
        QTableWidgetItem *__qtablewidgetitem95 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(41, __qtablewidgetitem95);
        QTableWidgetItem *__qtablewidgetitem96 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(42, __qtablewidgetitem96);
        QTableWidgetItem *__qtablewidgetitem97 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(43, __qtablewidgetitem97);
        QTableWidgetItem *__qtablewidgetitem98 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(44, __qtablewidgetitem98);
        QTableWidgetItem *__qtablewidgetitem99 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(45, __qtablewidgetitem99);
        QTableWidgetItem *__qtablewidgetitem100 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(46, __qtablewidgetitem100);
        QTableWidgetItem *__qtablewidgetitem101 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(47, __qtablewidgetitem101);
        QTableWidgetItem *__qtablewidgetitem102 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(48, __qtablewidgetitem102);
        QTableWidgetItem *__qtablewidgetitem103 = new QTableWidgetItem();
        ListFluTable->setVerticalHeaderItem(49, __qtablewidgetitem103);
        ListFluTable->setObjectName(QString::fromUtf8("ListFluTable"));
        ListFluTable->setGeometry(QRect(12, 46, 221, 171));
        ListFluTable->setRowCount(50);
        ListFluTable->setColumnCount(2);

        vboxLayout17->addWidget(PrintFluSpecGroupBox);

        IPRIMARYGroupBox = new QGroupBox(IOtab);
        IPRIMARYGroupBox->setObjectName(QString::fromUtf8("IPRIMARYGroupBox"));
        QSizePolicy sizePolicy8(QSizePolicy::Preferred, QSizePolicy::Maximum);
        sizePolicy8.setHorizontalStretch(0);
        sizePolicy8.setVerticalStretch(0);
        sizePolicy8.setHeightForWidth(IPRIMARYGroupBox->sizePolicy().hasHeightForWidth());
        IPRIMARYGroupBox->setSizePolicy(sizePolicy8);
        vboxLayout18 = new QVBoxLayout(IPRIMARYGroupBox);
        vboxLayout18->setSpacing(6);
        vboxLayout18->setContentsMargins(11, 11, 11, 11);
        vboxLayout18->setObjectName(QString::fromUtf8("vboxLayout18"));
        IPRIMARYComboBox = new QComboBox(IPRIMARYGroupBox);
        IPRIMARYComboBox->addItem(QString());
        IPRIMARYComboBox->addItem(QString());
        IPRIMARYComboBox->addItem(QString());
        IPRIMARYComboBox->addItem(QString());
        IPRIMARYComboBox->addItem(QString());
        IPRIMARYComboBox->setObjectName(QString::fromUtf8("IPRIMARYComboBox"));

        vboxLayout18->addWidget(IPRIMARYComboBox);


        vboxLayout17->addWidget(IPRIMARYGroupBox);

        sloteFluGroupBox = new QGroupBox(IOtab);
        sloteFluGroupBox->setObjectName(QString::fromUtf8("sloteFluGroupBox"));
        sloteFluLabel = new QLabel(sloteFluGroupBox);
        sloteFluLabel->setObjectName(QString::fromUtf8("sloteFluLabel"));
        sloteFluLabel->setGeometry(QRect(21, 21, 70, 28));
        sloteFluLabel->setWordWrap(false);
        sloteFluEdit = new QLineEdit(sloteFluGroupBox);
        sloteFluEdit->setObjectName(QString::fromUtf8("sloteFluEdit"));
        sloteFluEdit->setGeometry(QRect(93, 25, 90, 20));
        sloteFluTable = new QTableWidget(sloteFluGroupBox);
        if (sloteFluTable->columnCount() < 1)
            sloteFluTable->setColumnCount(1);
        QTableWidgetItem *__qtablewidgetitem104 = new QTableWidgetItem();
        sloteFluTable->setHorizontalHeaderItem(0, __qtablewidgetitem104);
        if (sloteFluTable->rowCount() < 200)
            sloteFluTable->setRowCount(200);
        QTableWidgetItem *__qtablewidgetitem105 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(0, __qtablewidgetitem105);
        QTableWidgetItem *__qtablewidgetitem106 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(1, __qtablewidgetitem106);
        QTableWidgetItem *__qtablewidgetitem107 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(2, __qtablewidgetitem107);
        QTableWidgetItem *__qtablewidgetitem108 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(3, __qtablewidgetitem108);
        QTableWidgetItem *__qtablewidgetitem109 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(4, __qtablewidgetitem109);
        QTableWidgetItem *__qtablewidgetitem110 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(5, __qtablewidgetitem110);
        QTableWidgetItem *__qtablewidgetitem111 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(6, __qtablewidgetitem111);
        QTableWidgetItem *__qtablewidgetitem112 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(7, __qtablewidgetitem112);
        QTableWidgetItem *__qtablewidgetitem113 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(8, __qtablewidgetitem113);
        QTableWidgetItem *__qtablewidgetitem114 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(9, __qtablewidgetitem114);
        QTableWidgetItem *__qtablewidgetitem115 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(10, __qtablewidgetitem115);
        QTableWidgetItem *__qtablewidgetitem116 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(11, __qtablewidgetitem116);
        QTableWidgetItem *__qtablewidgetitem117 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(12, __qtablewidgetitem117);
        QTableWidgetItem *__qtablewidgetitem118 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(13, __qtablewidgetitem118);
        QTableWidgetItem *__qtablewidgetitem119 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(14, __qtablewidgetitem119);
        QTableWidgetItem *__qtablewidgetitem120 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(15, __qtablewidgetitem120);
        QTableWidgetItem *__qtablewidgetitem121 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(16, __qtablewidgetitem121);
        QTableWidgetItem *__qtablewidgetitem122 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(17, __qtablewidgetitem122);
        QTableWidgetItem *__qtablewidgetitem123 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(18, __qtablewidgetitem123);
        QTableWidgetItem *__qtablewidgetitem124 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(19, __qtablewidgetitem124);
        QTableWidgetItem *__qtablewidgetitem125 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(20, __qtablewidgetitem125);
        QTableWidgetItem *__qtablewidgetitem126 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(21, __qtablewidgetitem126);
        QTableWidgetItem *__qtablewidgetitem127 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(22, __qtablewidgetitem127);
        QTableWidgetItem *__qtablewidgetitem128 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(23, __qtablewidgetitem128);
        QTableWidgetItem *__qtablewidgetitem129 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(24, __qtablewidgetitem129);
        QTableWidgetItem *__qtablewidgetitem130 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(25, __qtablewidgetitem130);
        QTableWidgetItem *__qtablewidgetitem131 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(26, __qtablewidgetitem131);
        QTableWidgetItem *__qtablewidgetitem132 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(27, __qtablewidgetitem132);
        QTableWidgetItem *__qtablewidgetitem133 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(28, __qtablewidgetitem133);
        QTableWidgetItem *__qtablewidgetitem134 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(29, __qtablewidgetitem134);
        QTableWidgetItem *__qtablewidgetitem135 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(30, __qtablewidgetitem135);
        QTableWidgetItem *__qtablewidgetitem136 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(31, __qtablewidgetitem136);
        QTableWidgetItem *__qtablewidgetitem137 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(32, __qtablewidgetitem137);
        QTableWidgetItem *__qtablewidgetitem138 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(33, __qtablewidgetitem138);
        QTableWidgetItem *__qtablewidgetitem139 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(34, __qtablewidgetitem139);
        QTableWidgetItem *__qtablewidgetitem140 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(35, __qtablewidgetitem140);
        QTableWidgetItem *__qtablewidgetitem141 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(36, __qtablewidgetitem141);
        QTableWidgetItem *__qtablewidgetitem142 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(37, __qtablewidgetitem142);
        QTableWidgetItem *__qtablewidgetitem143 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(38, __qtablewidgetitem143);
        QTableWidgetItem *__qtablewidgetitem144 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(39, __qtablewidgetitem144);
        QTableWidgetItem *__qtablewidgetitem145 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(40, __qtablewidgetitem145);
        QTableWidgetItem *__qtablewidgetitem146 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(41, __qtablewidgetitem146);
        QTableWidgetItem *__qtablewidgetitem147 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(42, __qtablewidgetitem147);
        QTableWidgetItem *__qtablewidgetitem148 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(43, __qtablewidgetitem148);
        QTableWidgetItem *__qtablewidgetitem149 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(44, __qtablewidgetitem149);
        QTableWidgetItem *__qtablewidgetitem150 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(45, __qtablewidgetitem150);
        QTableWidgetItem *__qtablewidgetitem151 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(46, __qtablewidgetitem151);
        QTableWidgetItem *__qtablewidgetitem152 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(47, __qtablewidgetitem152);
        QTableWidgetItem *__qtablewidgetitem153 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(48, __qtablewidgetitem153);
        QTableWidgetItem *__qtablewidgetitem154 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(49, __qtablewidgetitem154);
        QTableWidgetItem *__qtablewidgetitem155 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(50, __qtablewidgetitem155);
        QTableWidgetItem *__qtablewidgetitem156 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(51, __qtablewidgetitem156);
        QTableWidgetItem *__qtablewidgetitem157 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(52, __qtablewidgetitem157);
        QTableWidgetItem *__qtablewidgetitem158 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(53, __qtablewidgetitem158);
        QTableWidgetItem *__qtablewidgetitem159 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(54, __qtablewidgetitem159);
        QTableWidgetItem *__qtablewidgetitem160 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(55, __qtablewidgetitem160);
        QTableWidgetItem *__qtablewidgetitem161 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(56, __qtablewidgetitem161);
        QTableWidgetItem *__qtablewidgetitem162 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(57, __qtablewidgetitem162);
        QTableWidgetItem *__qtablewidgetitem163 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(58, __qtablewidgetitem163);
        QTableWidgetItem *__qtablewidgetitem164 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(59, __qtablewidgetitem164);
        QTableWidgetItem *__qtablewidgetitem165 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(60, __qtablewidgetitem165);
        QTableWidgetItem *__qtablewidgetitem166 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(61, __qtablewidgetitem166);
        QTableWidgetItem *__qtablewidgetitem167 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(62, __qtablewidgetitem167);
        QTableWidgetItem *__qtablewidgetitem168 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(63, __qtablewidgetitem168);
        QTableWidgetItem *__qtablewidgetitem169 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(64, __qtablewidgetitem169);
        QTableWidgetItem *__qtablewidgetitem170 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(65, __qtablewidgetitem170);
        QTableWidgetItem *__qtablewidgetitem171 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(66, __qtablewidgetitem171);
        QTableWidgetItem *__qtablewidgetitem172 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(67, __qtablewidgetitem172);
        QTableWidgetItem *__qtablewidgetitem173 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(68, __qtablewidgetitem173);
        QTableWidgetItem *__qtablewidgetitem174 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(69, __qtablewidgetitem174);
        QTableWidgetItem *__qtablewidgetitem175 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(70, __qtablewidgetitem175);
        QTableWidgetItem *__qtablewidgetitem176 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(71, __qtablewidgetitem176);
        QTableWidgetItem *__qtablewidgetitem177 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(72, __qtablewidgetitem177);
        QTableWidgetItem *__qtablewidgetitem178 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(73, __qtablewidgetitem178);
        QTableWidgetItem *__qtablewidgetitem179 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(74, __qtablewidgetitem179);
        QTableWidgetItem *__qtablewidgetitem180 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(75, __qtablewidgetitem180);
        QTableWidgetItem *__qtablewidgetitem181 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(76, __qtablewidgetitem181);
        QTableWidgetItem *__qtablewidgetitem182 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(77, __qtablewidgetitem182);
        QTableWidgetItem *__qtablewidgetitem183 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(78, __qtablewidgetitem183);
        QTableWidgetItem *__qtablewidgetitem184 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(79, __qtablewidgetitem184);
        QTableWidgetItem *__qtablewidgetitem185 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(80, __qtablewidgetitem185);
        QTableWidgetItem *__qtablewidgetitem186 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(81, __qtablewidgetitem186);
        QTableWidgetItem *__qtablewidgetitem187 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(82, __qtablewidgetitem187);
        QTableWidgetItem *__qtablewidgetitem188 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(83, __qtablewidgetitem188);
        QTableWidgetItem *__qtablewidgetitem189 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(84, __qtablewidgetitem189);
        QTableWidgetItem *__qtablewidgetitem190 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(85, __qtablewidgetitem190);
        QTableWidgetItem *__qtablewidgetitem191 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(86, __qtablewidgetitem191);
        QTableWidgetItem *__qtablewidgetitem192 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(87, __qtablewidgetitem192);
        QTableWidgetItem *__qtablewidgetitem193 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(88, __qtablewidgetitem193);
        QTableWidgetItem *__qtablewidgetitem194 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(89, __qtablewidgetitem194);
        QTableWidgetItem *__qtablewidgetitem195 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(90, __qtablewidgetitem195);
        QTableWidgetItem *__qtablewidgetitem196 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(91, __qtablewidgetitem196);
        QTableWidgetItem *__qtablewidgetitem197 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(92, __qtablewidgetitem197);
        QTableWidgetItem *__qtablewidgetitem198 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(93, __qtablewidgetitem198);
        QTableWidgetItem *__qtablewidgetitem199 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(94, __qtablewidgetitem199);
        QTableWidgetItem *__qtablewidgetitem200 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(95, __qtablewidgetitem200);
        QTableWidgetItem *__qtablewidgetitem201 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(96, __qtablewidgetitem201);
        QTableWidgetItem *__qtablewidgetitem202 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(97, __qtablewidgetitem202);
        QTableWidgetItem *__qtablewidgetitem203 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(98, __qtablewidgetitem203);
        QTableWidgetItem *__qtablewidgetitem204 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(99, __qtablewidgetitem204);
        QTableWidgetItem *__qtablewidgetitem205 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(100, __qtablewidgetitem205);
        QTableWidgetItem *__qtablewidgetitem206 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(101, __qtablewidgetitem206);
        QTableWidgetItem *__qtablewidgetitem207 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(102, __qtablewidgetitem207);
        QTableWidgetItem *__qtablewidgetitem208 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(103, __qtablewidgetitem208);
        QTableWidgetItem *__qtablewidgetitem209 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(104, __qtablewidgetitem209);
        QTableWidgetItem *__qtablewidgetitem210 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(105, __qtablewidgetitem210);
        QTableWidgetItem *__qtablewidgetitem211 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(106, __qtablewidgetitem211);
        QTableWidgetItem *__qtablewidgetitem212 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(107, __qtablewidgetitem212);
        QTableWidgetItem *__qtablewidgetitem213 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(108, __qtablewidgetitem213);
        QTableWidgetItem *__qtablewidgetitem214 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(109, __qtablewidgetitem214);
        QTableWidgetItem *__qtablewidgetitem215 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(110, __qtablewidgetitem215);
        QTableWidgetItem *__qtablewidgetitem216 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(111, __qtablewidgetitem216);
        QTableWidgetItem *__qtablewidgetitem217 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(112, __qtablewidgetitem217);
        QTableWidgetItem *__qtablewidgetitem218 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(113, __qtablewidgetitem218);
        QTableWidgetItem *__qtablewidgetitem219 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(114, __qtablewidgetitem219);
        QTableWidgetItem *__qtablewidgetitem220 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(115, __qtablewidgetitem220);
        QTableWidgetItem *__qtablewidgetitem221 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(116, __qtablewidgetitem221);
        QTableWidgetItem *__qtablewidgetitem222 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(117, __qtablewidgetitem222);
        QTableWidgetItem *__qtablewidgetitem223 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(118, __qtablewidgetitem223);
        QTableWidgetItem *__qtablewidgetitem224 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(119, __qtablewidgetitem224);
        QTableWidgetItem *__qtablewidgetitem225 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(120, __qtablewidgetitem225);
        QTableWidgetItem *__qtablewidgetitem226 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(121, __qtablewidgetitem226);
        QTableWidgetItem *__qtablewidgetitem227 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(122, __qtablewidgetitem227);
        QTableWidgetItem *__qtablewidgetitem228 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(123, __qtablewidgetitem228);
        QTableWidgetItem *__qtablewidgetitem229 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(124, __qtablewidgetitem229);
        QTableWidgetItem *__qtablewidgetitem230 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(125, __qtablewidgetitem230);
        QTableWidgetItem *__qtablewidgetitem231 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(126, __qtablewidgetitem231);
        QTableWidgetItem *__qtablewidgetitem232 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(127, __qtablewidgetitem232);
        QTableWidgetItem *__qtablewidgetitem233 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(128, __qtablewidgetitem233);
        QTableWidgetItem *__qtablewidgetitem234 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(129, __qtablewidgetitem234);
        QTableWidgetItem *__qtablewidgetitem235 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(130, __qtablewidgetitem235);
        QTableWidgetItem *__qtablewidgetitem236 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(131, __qtablewidgetitem236);
        QTableWidgetItem *__qtablewidgetitem237 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(132, __qtablewidgetitem237);
        QTableWidgetItem *__qtablewidgetitem238 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(133, __qtablewidgetitem238);
        QTableWidgetItem *__qtablewidgetitem239 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(134, __qtablewidgetitem239);
        QTableWidgetItem *__qtablewidgetitem240 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(135, __qtablewidgetitem240);
        QTableWidgetItem *__qtablewidgetitem241 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(136, __qtablewidgetitem241);
        QTableWidgetItem *__qtablewidgetitem242 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(137, __qtablewidgetitem242);
        QTableWidgetItem *__qtablewidgetitem243 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(138, __qtablewidgetitem243);
        QTableWidgetItem *__qtablewidgetitem244 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(139, __qtablewidgetitem244);
        QTableWidgetItem *__qtablewidgetitem245 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(140, __qtablewidgetitem245);
        QTableWidgetItem *__qtablewidgetitem246 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(141, __qtablewidgetitem246);
        QTableWidgetItem *__qtablewidgetitem247 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(142, __qtablewidgetitem247);
        QTableWidgetItem *__qtablewidgetitem248 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(143, __qtablewidgetitem248);
        QTableWidgetItem *__qtablewidgetitem249 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(144, __qtablewidgetitem249);
        QTableWidgetItem *__qtablewidgetitem250 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(145, __qtablewidgetitem250);
        QTableWidgetItem *__qtablewidgetitem251 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(146, __qtablewidgetitem251);
        QTableWidgetItem *__qtablewidgetitem252 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(147, __qtablewidgetitem252);
        QTableWidgetItem *__qtablewidgetitem253 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(148, __qtablewidgetitem253);
        QTableWidgetItem *__qtablewidgetitem254 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(149, __qtablewidgetitem254);
        QTableWidgetItem *__qtablewidgetitem255 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(150, __qtablewidgetitem255);
        QTableWidgetItem *__qtablewidgetitem256 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(151, __qtablewidgetitem256);
        QTableWidgetItem *__qtablewidgetitem257 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(152, __qtablewidgetitem257);
        QTableWidgetItem *__qtablewidgetitem258 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(153, __qtablewidgetitem258);
        QTableWidgetItem *__qtablewidgetitem259 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(154, __qtablewidgetitem259);
        QTableWidgetItem *__qtablewidgetitem260 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(155, __qtablewidgetitem260);
        QTableWidgetItem *__qtablewidgetitem261 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(156, __qtablewidgetitem261);
        QTableWidgetItem *__qtablewidgetitem262 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(157, __qtablewidgetitem262);
        QTableWidgetItem *__qtablewidgetitem263 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(158, __qtablewidgetitem263);
        QTableWidgetItem *__qtablewidgetitem264 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(159, __qtablewidgetitem264);
        QTableWidgetItem *__qtablewidgetitem265 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(160, __qtablewidgetitem265);
        QTableWidgetItem *__qtablewidgetitem266 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(161, __qtablewidgetitem266);
        QTableWidgetItem *__qtablewidgetitem267 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(162, __qtablewidgetitem267);
        QTableWidgetItem *__qtablewidgetitem268 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(163, __qtablewidgetitem268);
        QTableWidgetItem *__qtablewidgetitem269 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(164, __qtablewidgetitem269);
        QTableWidgetItem *__qtablewidgetitem270 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(165, __qtablewidgetitem270);
        QTableWidgetItem *__qtablewidgetitem271 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(166, __qtablewidgetitem271);
        QTableWidgetItem *__qtablewidgetitem272 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(167, __qtablewidgetitem272);
        QTableWidgetItem *__qtablewidgetitem273 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(168, __qtablewidgetitem273);
        QTableWidgetItem *__qtablewidgetitem274 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(169, __qtablewidgetitem274);
        QTableWidgetItem *__qtablewidgetitem275 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(170, __qtablewidgetitem275);
        QTableWidgetItem *__qtablewidgetitem276 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(171, __qtablewidgetitem276);
        QTableWidgetItem *__qtablewidgetitem277 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(172, __qtablewidgetitem277);
        QTableWidgetItem *__qtablewidgetitem278 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(173, __qtablewidgetitem278);
        QTableWidgetItem *__qtablewidgetitem279 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(174, __qtablewidgetitem279);
        QTableWidgetItem *__qtablewidgetitem280 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(175, __qtablewidgetitem280);
        QTableWidgetItem *__qtablewidgetitem281 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(176, __qtablewidgetitem281);
        QTableWidgetItem *__qtablewidgetitem282 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(177, __qtablewidgetitem282);
        QTableWidgetItem *__qtablewidgetitem283 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(178, __qtablewidgetitem283);
        QTableWidgetItem *__qtablewidgetitem284 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(179, __qtablewidgetitem284);
        QTableWidgetItem *__qtablewidgetitem285 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(180, __qtablewidgetitem285);
        QTableWidgetItem *__qtablewidgetitem286 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(181, __qtablewidgetitem286);
        QTableWidgetItem *__qtablewidgetitem287 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(182, __qtablewidgetitem287);
        QTableWidgetItem *__qtablewidgetitem288 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(183, __qtablewidgetitem288);
        QTableWidgetItem *__qtablewidgetitem289 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(184, __qtablewidgetitem289);
        QTableWidgetItem *__qtablewidgetitem290 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(185, __qtablewidgetitem290);
        QTableWidgetItem *__qtablewidgetitem291 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(186, __qtablewidgetitem291);
        QTableWidgetItem *__qtablewidgetitem292 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(187, __qtablewidgetitem292);
        QTableWidgetItem *__qtablewidgetitem293 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(188, __qtablewidgetitem293);
        QTableWidgetItem *__qtablewidgetitem294 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(189, __qtablewidgetitem294);
        QTableWidgetItem *__qtablewidgetitem295 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(190, __qtablewidgetitem295);
        QTableWidgetItem *__qtablewidgetitem296 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(191, __qtablewidgetitem296);
        QTableWidgetItem *__qtablewidgetitem297 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(192, __qtablewidgetitem297);
        QTableWidgetItem *__qtablewidgetitem298 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(193, __qtablewidgetitem298);
        QTableWidgetItem *__qtablewidgetitem299 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(194, __qtablewidgetitem299);
        QTableWidgetItem *__qtablewidgetitem300 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(195, __qtablewidgetitem300);
        QTableWidgetItem *__qtablewidgetitem301 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(196, __qtablewidgetitem301);
        QTableWidgetItem *__qtablewidgetitem302 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(197, __qtablewidgetitem302);
        QTableWidgetItem *__qtablewidgetitem303 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(198, __qtablewidgetitem303);
        QTableWidgetItem *__qtablewidgetitem304 = new QTableWidgetItem();
        sloteFluTable->setVerticalHeaderItem(199, __qtablewidgetitem304);
        sloteFluTable->setObjectName(QString::fromUtf8("sloteFluTable"));
        sloteFluTable->setGeometry(QRect(23, 50, 150, 151));
        sloteFluTable->setRowCount(200);
        sloteFluTable->setColumnCount(1);

        vboxLayout17->addWidget(sloteFluGroupBox);


        hboxLayout7->addLayout(vboxLayout17);


        hboxLayout6->addLayout(hboxLayout7);

        TabWidgetRZ->addTab(IOtab, QString());
        MCtab = new QWidget();
        MCtab->setObjectName(QString::fromUtf8("MCtab"));
        Layout46 = new QWidget(MCtab);
        Layout46->setObjectName(QString::fromUtf8("Layout46"));
        Layout46->setGeometry(QRect(19, 15, 321, 116));
        vboxLayout19 = new QVBoxLayout(Layout46);
        vboxLayout19->setSpacing(6);
        vboxLayout19->setContentsMargins(11, 11, 11, 11);
        vboxLayout19->setObjectName(QString::fromUtf8("vboxLayout19"));
        vboxLayout19->setContentsMargins(0, 0, 0, 0);
        hboxLayout16 = new QHBoxLayout();
        hboxLayout16->setSpacing(6);
        hboxLayout16->setObjectName(QString::fromUtf8("hboxLayout16"));
        ncaseLabel = new QLabel(Layout46);
        ncaseLabel->setObjectName(QString::fromUtf8("ncaseLabel"));
        ncaseLabel->setWordWrap(false);

        hboxLayout16->addWidget(ncaseLabel);

        Spacer6_2 = new QSpacerItem(20, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout16->addItem(Spacer6_2);

        ncaseEdit = new QLineEdit(Layout46);
        ncaseEdit->setObjectName(QString::fromUtf8("ncaseEdit"));

        hboxLayout16->addWidget(ncaseEdit);


        vboxLayout19->addLayout(hboxLayout16);

        hboxLayout17 = new QHBoxLayout();
        hboxLayout17->setSpacing(6);
        hboxLayout17->setObjectName(QString::fromUtf8("hboxLayout17"));
        maxCPULabel = new QLabel(Layout46);
        maxCPULabel->setObjectName(QString::fromUtf8("maxCPULabel"));
        maxCPULabel->setWordWrap(false);

        hboxLayout17->addWidget(maxCPULabel);

        Spacer9 = new QSpacerItem(20, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout17->addItem(Spacer9);

        maxCPUEdit = new QLineEdit(Layout46);
        maxCPUEdit->setObjectName(QString::fromUtf8("maxCPUEdit"));

        hboxLayout17->addWidget(maxCPUEdit);


        vboxLayout19->addLayout(hboxLayout17);

        hboxLayout18 = new QHBoxLayout();
        hboxLayout18->setSpacing(6);
        hboxLayout18->setObjectName(QString::fromUtf8("hboxLayout18"));
        statLabel = new QLabel(Layout46);
        statLabel->setObjectName(QString::fromUtf8("statLabel"));
        statLabel->setWordWrap(false);

        hboxLayout18->addWidget(statLabel);

        Spacer8_2 = new QSpacerItem(20, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout18->addItem(Spacer8_2);

        statEdit = new QLineEdit(Layout46);
        statEdit->setObjectName(QString::fromUtf8("statEdit"));

        hboxLayout18->addWidget(statEdit);


        vboxLayout19->addLayout(hboxLayout18);

        kermaGroupBox = new QGroupBox(MCtab);
        kermaGroupBox->setObjectName(QString::fromUtf8("kermaGroupBox"));
        kermaGroupBox->setEnabled(false);
        kermaGroupBox->setGeometry(QRect(18, 281, 180, 40));
        kermaCheckBox = new QCheckBox(kermaGroupBox);
        kermaCheckBox->setObjectName(QString::fromUtf8("kermaCheckBox"));
        kermaCheckBox->setGeometry(QRect(20, 10, 140, 19));
        randGroupBox = new QGroupBox(MCtab);
        randGroupBox->setObjectName(QString::fromUtf8("randGroupBox"));
        randGroupBox->setGeometry(QRect(18, 150, 318, 116));
        vboxLayout20 = new QVBoxLayout(randGroupBox);
        vboxLayout20->setSpacing(6);
        vboxLayout20->setContentsMargins(11, 11, 11, 11);
        vboxLayout20->setObjectName(QString::fromUtf8("vboxLayout20"));
        hboxLayout19 = new QHBoxLayout();
        hboxLayout19->setSpacing(6);
        hboxLayout19->setObjectName(QString::fromUtf8("hboxLayout19"));
        rand1Label = new QLabel(randGroupBox);
        rand1Label->setObjectName(QString::fromUtf8("rand1Label"));
        rand1Label->setWordWrap(false);

        hboxLayout19->addWidget(rand1Label);

        Spacer4_2 = new QSpacerItem(20, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout19->addItem(Spacer4_2);

        rand1SpinBox = new QSpinBox(randGroupBox);
        rand1SpinBox->setObjectName(QString::fromUtf8("rand1SpinBox"));
        QSizePolicy sizePolicy9(QSizePolicy::Minimum, QSizePolicy::Minimum);
        sizePolicy9.setHorizontalStretch(0);
        sizePolicy9.setVerticalStretch(0);
        sizePolicy9.setHeightForWidth(rand1SpinBox->sizePolicy().hasHeightForWidth());
        rand1SpinBox->setSizePolicy(sizePolicy9);
        rand1SpinBox->setMinimumSize(QSize(100, 0));
        rand1SpinBox->setMinimum(1);
        rand1SpinBox->setMaximum(31328);
        rand1SpinBox->setValue(1);

        hboxLayout19->addWidget(rand1SpinBox);


        vboxLayout20->addLayout(hboxLayout19);

        hboxLayout20 = new QHBoxLayout();
        hboxLayout20->setSpacing(6);
        hboxLayout20->setObjectName(QString::fromUtf8("hboxLayout20"));
        rand2Label = new QLabel(randGroupBox);
        rand2Label->setObjectName(QString::fromUtf8("rand2Label"));
        rand2Label->setWordWrap(false);

        hboxLayout20->addWidget(rand2Label);

        Spacer5_2 = new QSpacerItem(20, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout20->addItem(Spacer5_2);

        rand2SpinBox = new QSpinBox(randGroupBox);
        rand2SpinBox->setObjectName(QString::fromUtf8("rand2SpinBox"));
        rand2SpinBox->setMinimumSize(QSize(100, 0));
        rand2SpinBox->setMinimum(1);
        rand2SpinBox->setMaximum(1073741824);
        rand2SpinBox->setValue(33);

        hboxLayout20->addWidget(rand2SpinBox);


        vboxLayout20->addLayout(hboxLayout20);

        photregGroupBox = new QGroupBox(MCtab);
        photregGroupBox->setObjectName(QString::fromUtf8("photregGroupBox"));
        photregGroupBox->setGeometry(QRect(20, 350, 242, 52));
        hboxLayout21 = new QHBoxLayout(photregGroupBox);
        hboxLayout21->setSpacing(6);
        hboxLayout21->setContentsMargins(11, 11, 11, 11);
        hboxLayout21->setObjectName(QString::fromUtf8("hboxLayout21"));
        photregCheckBox = new QCheckBox(photregGroupBox);
        photregCheckBox->setObjectName(QString::fromUtf8("photregCheckBox"));

        hboxLayout21->addWidget(photregCheckBox);

        layout157 = new QWidget(MCtab);
        layout157->setObjectName(QString::fromUtf8("layout157"));
        layout157->setGeometry(QRect(430, 10, 321, 521));
        vboxLayout21 = new QVBoxLayout(layout157);
        vboxLayout21->setSpacing(6);
        vboxLayout21->setContentsMargins(11, 11, 11, 11);
        vboxLayout21->setObjectName(QString::fromUtf8("vboxLayout21"));
        vboxLayout21->setContentsMargins(0, 0, 0, 0);
        ifullGroupBox = new QGroupBox(layout157);
        ifullGroupBox->setObjectName(QString::fromUtf8("ifullGroupBox"));
        sizePolicy8.setHeightForWidth(ifullGroupBox->sizePolicy().hasHeightForWidth());
        ifullGroupBox->setSizePolicy(sizePolicy8);
        vboxLayout22 = new QVBoxLayout(ifullGroupBox);
        vboxLayout22->setSpacing(6);
        vboxLayout22->setContentsMargins(11, 11, 11, 11);
        vboxLayout22->setObjectName(QString::fromUtf8("vboxLayout22"));
        hboxLayout22 = new QHBoxLayout();
        hboxLayout22->setSpacing(6);
        hboxLayout22->setObjectName(QString::fromUtf8("hboxLayout22"));
        ifullLabel = new QLabel(ifullGroupBox);
        ifullLabel->setObjectName(QString::fromUtf8("ifullLabel"));
        ifullLabel->setWordWrap(false);

        hboxLayout22->addWidget(ifullLabel);

        Spacer77 = new QSpacerItem(20, 20, QSizePolicy::Fixed, QSizePolicy::Minimum);

        hboxLayout22->addItem(Spacer77);

        ifullComboBox = new QComboBox(ifullGroupBox);
        ifullComboBox->addItem(QString());
        ifullComboBox->addItem(QString());
        ifullComboBox->addItem(QString());
        ifullComboBox->addItem(QString());
        ifullComboBox->setObjectName(QString::fromUtf8("ifullComboBox"));
        sizePolicy4.setHeightForWidth(ifullComboBox->sizePolicy().hasHeightForWidth());
        ifullComboBox->setSizePolicy(sizePolicy4);

        hboxLayout22->addWidget(ifullComboBox);


        vboxLayout22->addLayout(hboxLayout22);


        vboxLayout21->addWidget(ifullGroupBox);

        phdGroupBox = new QGroupBox(layout157);
        phdGroupBox->setObjectName(QString::fromUtf8("phdGroupBox"));
        phdGroupBox->setEnabled(false);
        Layout30 = new QWidget(phdGroupBox);
        Layout30->setObjectName(QString::fromUtf8("Layout30"));
        Layout30->setGeometry(QRect(60, 30, 201, 70));
        vboxLayout23 = new QVBoxLayout(Layout30);
        vboxLayout23->setSpacing(6);
        vboxLayout23->setContentsMargins(11, 11, 11, 11);
        vboxLayout23->setObjectName(QString::fromUtf8("vboxLayout23"));
        vboxLayout23->setContentsMargins(0, 0, 0, 0);
        hboxLayout23 = new QHBoxLayout();
        hboxLayout23->setSpacing(6);
        hboxLayout23->setObjectName(QString::fromUtf8("hboxLayout23"));
        SLOTELabel = new QLabel(Layout30);
        SLOTELabel->setObjectName(QString::fromUtf8("SLOTELabel"));
        SLOTELabel->setWordWrap(false);

        hboxLayout23->addWidget(SLOTELabel);

        Spacer1_2 = new QSpacerItem(20, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout23->addItem(Spacer1_2);

        SLOTEEdit = new QLineEdit(Layout30);
        SLOTEEdit->setObjectName(QString::fromUtf8("SLOTEEdit"));

        hboxLayout23->addWidget(SLOTEEdit);


        vboxLayout23->addLayout(hboxLayout23);

        hboxLayout24 = new QHBoxLayout();
        hboxLayout24->setSpacing(6);
        hboxLayout24->setObjectName(QString::fromUtf8("hboxLayout24"));
        DELTAELabel = new QLabel(Layout30);
        DELTAELabel->setObjectName(QString::fromUtf8("DELTAELabel"));
        DELTAELabel->setWordWrap(false);

        hboxLayout24->addWidget(DELTAELabel);

        Spacer2_2 = new QSpacerItem(20, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout24->addItem(Spacer2_2);

        DELTAEEdit = new QLineEdit(Layout30);
        DELTAEEdit->setObjectName(QString::fromUtf8("DELTAEEdit"));

        hboxLayout24->addWidget(DELTAEEdit);


        vboxLayout23->addLayout(hboxLayout24);

        phdTable = new QTableWidget(phdGroupBox);
        if (phdTable->columnCount() < 2)
            phdTable->setColumnCount(2);
        QTableWidgetItem *__qtablewidgetitem305 = new QTableWidgetItem();
        phdTable->setHorizontalHeaderItem(0, __qtablewidgetitem305);
        QTableWidgetItem *__qtablewidgetitem306 = new QTableWidgetItem();
        phdTable->setHorizontalHeaderItem(1, __qtablewidgetitem306);
        if (phdTable->rowCount() < 50)
            phdTable->setRowCount(50);
        QTableWidgetItem *__qtablewidgetitem307 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(0, __qtablewidgetitem307);
        QTableWidgetItem *__qtablewidgetitem308 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(1, __qtablewidgetitem308);
        QTableWidgetItem *__qtablewidgetitem309 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(2, __qtablewidgetitem309);
        QTableWidgetItem *__qtablewidgetitem310 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(3, __qtablewidgetitem310);
        QTableWidgetItem *__qtablewidgetitem311 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(4, __qtablewidgetitem311);
        QTableWidgetItem *__qtablewidgetitem312 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(5, __qtablewidgetitem312);
        QTableWidgetItem *__qtablewidgetitem313 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(6, __qtablewidgetitem313);
        QTableWidgetItem *__qtablewidgetitem314 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(7, __qtablewidgetitem314);
        QTableWidgetItem *__qtablewidgetitem315 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(8, __qtablewidgetitem315);
        QTableWidgetItem *__qtablewidgetitem316 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(9, __qtablewidgetitem316);
        QTableWidgetItem *__qtablewidgetitem317 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(10, __qtablewidgetitem317);
        QTableWidgetItem *__qtablewidgetitem318 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(11, __qtablewidgetitem318);
        QTableWidgetItem *__qtablewidgetitem319 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(12, __qtablewidgetitem319);
        QTableWidgetItem *__qtablewidgetitem320 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(13, __qtablewidgetitem320);
        QTableWidgetItem *__qtablewidgetitem321 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(14, __qtablewidgetitem321);
        QTableWidgetItem *__qtablewidgetitem322 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(15, __qtablewidgetitem322);
        QTableWidgetItem *__qtablewidgetitem323 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(16, __qtablewidgetitem323);
        QTableWidgetItem *__qtablewidgetitem324 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(17, __qtablewidgetitem324);
        QTableWidgetItem *__qtablewidgetitem325 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(18, __qtablewidgetitem325);
        QTableWidgetItem *__qtablewidgetitem326 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(19, __qtablewidgetitem326);
        QTableWidgetItem *__qtablewidgetitem327 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(20, __qtablewidgetitem327);
        QTableWidgetItem *__qtablewidgetitem328 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(21, __qtablewidgetitem328);
        QTableWidgetItem *__qtablewidgetitem329 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(22, __qtablewidgetitem329);
        QTableWidgetItem *__qtablewidgetitem330 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(23, __qtablewidgetitem330);
        QTableWidgetItem *__qtablewidgetitem331 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(24, __qtablewidgetitem331);
        QTableWidgetItem *__qtablewidgetitem332 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(25, __qtablewidgetitem332);
        QTableWidgetItem *__qtablewidgetitem333 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(26, __qtablewidgetitem333);
        QTableWidgetItem *__qtablewidgetitem334 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(27, __qtablewidgetitem334);
        QTableWidgetItem *__qtablewidgetitem335 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(28, __qtablewidgetitem335);
        QTableWidgetItem *__qtablewidgetitem336 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(29, __qtablewidgetitem336);
        QTableWidgetItem *__qtablewidgetitem337 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(30, __qtablewidgetitem337);
        QTableWidgetItem *__qtablewidgetitem338 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(31, __qtablewidgetitem338);
        QTableWidgetItem *__qtablewidgetitem339 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(32, __qtablewidgetitem339);
        QTableWidgetItem *__qtablewidgetitem340 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(33, __qtablewidgetitem340);
        QTableWidgetItem *__qtablewidgetitem341 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(34, __qtablewidgetitem341);
        QTableWidgetItem *__qtablewidgetitem342 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(35, __qtablewidgetitem342);
        QTableWidgetItem *__qtablewidgetitem343 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(36, __qtablewidgetitem343);
        QTableWidgetItem *__qtablewidgetitem344 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(37, __qtablewidgetitem344);
        QTableWidgetItem *__qtablewidgetitem345 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(38, __qtablewidgetitem345);
        QTableWidgetItem *__qtablewidgetitem346 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(39, __qtablewidgetitem346);
        QTableWidgetItem *__qtablewidgetitem347 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(40, __qtablewidgetitem347);
        QTableWidgetItem *__qtablewidgetitem348 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(41, __qtablewidgetitem348);
        QTableWidgetItem *__qtablewidgetitem349 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(42, __qtablewidgetitem349);
        QTableWidgetItem *__qtablewidgetitem350 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(43, __qtablewidgetitem350);
        QTableWidgetItem *__qtablewidgetitem351 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(44, __qtablewidgetitem351);
        QTableWidgetItem *__qtablewidgetitem352 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(45, __qtablewidgetitem352);
        QTableWidgetItem *__qtablewidgetitem353 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(46, __qtablewidgetitem353);
        QTableWidgetItem *__qtablewidgetitem354 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(47, __qtablewidgetitem354);
        QTableWidgetItem *__qtablewidgetitem355 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(48, __qtablewidgetitem355);
        QTableWidgetItem *__qtablewidgetitem356 = new QTableWidgetItem();
        phdTable->setVerticalHeaderItem(49, __qtablewidgetitem356);
        phdTable->setObjectName(QString::fromUtf8("phdTable"));
        phdTable->setGeometry(QRect(30, 110, 251, 311));
        phdTable->setRowCount(50);
        phdTable->setColumnCount(2);

        vboxLayout21->addWidget(phdGroupBox);

        TabWidgetRZ->addTab(MCtab, QString());
        Gtab = new QWidget();
        Gtab->setObjectName(QString::fromUtf8("Gtab"));
        gridLayout_24 = new QGridLayout(Gtab);
        gridLayout_24->setSpacing(6);
        gridLayout_24->setContentsMargins(11, 11, 11, 11);
        gridLayout_24->setObjectName(QString::fromUtf8("gridLayout_24"));
        gridLayout_23 = new QGridLayout();
        gridLayout_23->setSpacing(6);
        gridLayout_23->setObjectName(QString::fromUtf8("gridLayout_23"));
        horizontalLayout_20 = new QHBoxLayout();
        horizontalLayout_20->setSpacing(6);
        horizontalLayout_20->setObjectName(QString::fromUtf8("horizontalLayout_20"));
        inputmethButtonGroup = new QGroupBox(Gtab);
        inputmethButtonGroup->setObjectName(QString::fromUtf8("inputmethButtonGroup"));
        vboxLayout24 = new QVBoxLayout(inputmethButtonGroup);
        vboxLayout24->setSpacing(6);
        vboxLayout24->setContentsMargins(11, 11, 11, 11);
        vboxLayout24->setObjectName(QString::fromUtf8("vboxLayout24"));
        vboxLayout25 = new QVBoxLayout();
        vboxLayout25->setSpacing(6);
        vboxLayout25->setObjectName(QString::fromUtf8("vboxLayout25"));
        groupRadioButton = new QRadioButton(inputmethButtonGroup);
        groupRadioButton->setObjectName(QString::fromUtf8("groupRadioButton"));
        groupRadioButton->setChecked(true);

        vboxLayout25->addWidget(groupRadioButton);

        individualRadioButton = new QRadioButton(inputmethButtonGroup);
        individualRadioButton->setObjectName(QString::fromUtf8("individualRadioButton"));

        vboxLayout25->addWidget(individualRadioButton);

        cavityRadioButton = new QRadioButton(inputmethButtonGroup);
        cavityRadioButton->setObjectName(QString::fromUtf8("cavityRadioButton"));

        vboxLayout25->addWidget(cavityRadioButton);


        vboxLayout24->addLayout(vboxLayout25);


        horizontalLayout_20->addWidget(inputmethButtonGroup);

        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setSpacing(6);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        hboxLayout25 = new QHBoxLayout();
        hboxLayout25->setSpacing(6);
        hboxLayout25->setObjectName(QString::fromUtf8("hboxLayout25"));
        hboxLayout26 = new QHBoxLayout();
        hboxLayout26->setSpacing(6);
        hboxLayout26->setObjectName(QString::fromUtf8("hboxLayout26"));
        ZFaceLabel = new QLabel(Gtab);
        ZFaceLabel->setObjectName(QString::fromUtf8("ZFaceLabel"));
        ZFaceLabel->setWordWrap(false);

        hboxLayout26->addWidget(ZFaceLabel);

        ZFaceEdit = new QLineEdit(Gtab);
        ZFaceEdit->setObjectName(QString::fromUtf8("ZFaceEdit"));

        hboxLayout26->addWidget(ZFaceEdit);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout26->addItem(horizontalSpacer_3);


        hboxLayout25->addLayout(hboxLayout26);


        verticalLayout_3->addLayout(hboxLayout25);

        CavityInfoLabel = new QLabel(Gtab);
        CavityInfoLabel->setObjectName(QString::fromUtf8("CavityInfoLabel"));
        CavityInfoLabel->setWordWrap(false);

        verticalLayout_3->addWidget(CavityInfoLabel);


        horizontalLayout_20->addLayout(verticalLayout_3);


        gridLayout_23->addLayout(horizontalLayout_20, 0, 0, 1, 1);

        mediaGroupBox = new QGroupBox(Gtab);
        mediaGroupBox->setObjectName(QString::fromUtf8("mediaGroupBox"));
        mediaGroupBox->setMinimumSize(QSize(501, 0));
        verticalLayout_36 = new QVBoxLayout(mediaGroupBox);
        verticalLayout_36->setSpacing(6);
        verticalLayout_36->setContentsMargins(11, 11, 11, 11);
        verticalLayout_36->setObjectName(QString::fromUtf8("verticalLayout_36"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        mediaLabel = new QLabel(mediaGroupBox);
        mediaLabel->setObjectName(QString::fromUtf8("mediaLabel"));
        QSizePolicy sizePolicy10(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy10.setHorizontalStretch(1);
        sizePolicy10.setVerticalStretch(0);
        sizePolicy10.setHeightForWidth(mediaLabel->sizePolicy().hasHeightForWidth());
        mediaLabel->setSizePolicy(sizePolicy10);
        mediaLabel->setWordWrap(false);

        horizontalLayout->addWidget(mediaLabel);

        spacer42_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(spacer42_2);

        mediaComboBox = new QComboBox(mediaGroupBox);
        mediaComboBox->addItem(QString());
        mediaComboBox->addItem(QString());
        mediaComboBox->setObjectName(QString::fromUtf8("mediaComboBox"));
        QSizePolicy sizePolicy11(QSizePolicy::Expanding, QSizePolicy::Maximum);
        sizePolicy11.setHorizontalStretch(16);
        sizePolicy11.setVerticalStretch(0);
        sizePolicy11.setHeightForWidth(mediaComboBox->sizePolicy().hasHeightForWidth());
        mediaComboBox->setSizePolicy(sizePolicy11);
        mediaComboBox->setMinimumSize(QSize(130, 20));

        horizontalLayout->addWidget(mediaComboBox);


        verticalLayout_36->addLayout(horizontalLayout);

        mediaTable = new QTableWidget(mediaGroupBox);
        if (mediaTable->columnCount() < 3)
            mediaTable->setColumnCount(3);
        QTableWidgetItem *__qtablewidgetitem357 = new QTableWidgetItem();
        mediaTable->setHorizontalHeaderItem(0, __qtablewidgetitem357);
        QTableWidgetItem *__qtablewidgetitem358 = new QTableWidgetItem();
        mediaTable->setHorizontalHeaderItem(1, __qtablewidgetitem358);
        QTableWidgetItem *__qtablewidgetitem359 = new QTableWidgetItem();
        mediaTable->setHorizontalHeaderItem(2, __qtablewidgetitem359);
        if (mediaTable->rowCount() < 50)
            mediaTable->setRowCount(50);
        QTableWidgetItem *__qtablewidgetitem360 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(0, __qtablewidgetitem360);
        QTableWidgetItem *__qtablewidgetitem361 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(1, __qtablewidgetitem361);
        QTableWidgetItem *__qtablewidgetitem362 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(2, __qtablewidgetitem362);
        QTableWidgetItem *__qtablewidgetitem363 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(3, __qtablewidgetitem363);
        QTableWidgetItem *__qtablewidgetitem364 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(4, __qtablewidgetitem364);
        QTableWidgetItem *__qtablewidgetitem365 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(5, __qtablewidgetitem365);
        QTableWidgetItem *__qtablewidgetitem366 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(6, __qtablewidgetitem366);
        QTableWidgetItem *__qtablewidgetitem367 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(7, __qtablewidgetitem367);
        QTableWidgetItem *__qtablewidgetitem368 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(8, __qtablewidgetitem368);
        QTableWidgetItem *__qtablewidgetitem369 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(9, __qtablewidgetitem369);
        QTableWidgetItem *__qtablewidgetitem370 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(10, __qtablewidgetitem370);
        QTableWidgetItem *__qtablewidgetitem371 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(11, __qtablewidgetitem371);
        QTableWidgetItem *__qtablewidgetitem372 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(12, __qtablewidgetitem372);
        QTableWidgetItem *__qtablewidgetitem373 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(13, __qtablewidgetitem373);
        QTableWidgetItem *__qtablewidgetitem374 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(14, __qtablewidgetitem374);
        QTableWidgetItem *__qtablewidgetitem375 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(15, __qtablewidgetitem375);
        QTableWidgetItem *__qtablewidgetitem376 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(16, __qtablewidgetitem376);
        QTableWidgetItem *__qtablewidgetitem377 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(17, __qtablewidgetitem377);
        QTableWidgetItem *__qtablewidgetitem378 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(18, __qtablewidgetitem378);
        QTableWidgetItem *__qtablewidgetitem379 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(19, __qtablewidgetitem379);
        QTableWidgetItem *__qtablewidgetitem380 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(20, __qtablewidgetitem380);
        QTableWidgetItem *__qtablewidgetitem381 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(21, __qtablewidgetitem381);
        QTableWidgetItem *__qtablewidgetitem382 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(22, __qtablewidgetitem382);
        QTableWidgetItem *__qtablewidgetitem383 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(23, __qtablewidgetitem383);
        QTableWidgetItem *__qtablewidgetitem384 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(24, __qtablewidgetitem384);
        QTableWidgetItem *__qtablewidgetitem385 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(25, __qtablewidgetitem385);
        QTableWidgetItem *__qtablewidgetitem386 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(26, __qtablewidgetitem386);
        QTableWidgetItem *__qtablewidgetitem387 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(27, __qtablewidgetitem387);
        QTableWidgetItem *__qtablewidgetitem388 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(28, __qtablewidgetitem388);
        QTableWidgetItem *__qtablewidgetitem389 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(29, __qtablewidgetitem389);
        QTableWidgetItem *__qtablewidgetitem390 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(30, __qtablewidgetitem390);
        QTableWidgetItem *__qtablewidgetitem391 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(31, __qtablewidgetitem391);
        QTableWidgetItem *__qtablewidgetitem392 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(32, __qtablewidgetitem392);
        QTableWidgetItem *__qtablewidgetitem393 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(33, __qtablewidgetitem393);
        QTableWidgetItem *__qtablewidgetitem394 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(34, __qtablewidgetitem394);
        QTableWidgetItem *__qtablewidgetitem395 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(35, __qtablewidgetitem395);
        QTableWidgetItem *__qtablewidgetitem396 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(36, __qtablewidgetitem396);
        QTableWidgetItem *__qtablewidgetitem397 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(37, __qtablewidgetitem397);
        QTableWidgetItem *__qtablewidgetitem398 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(38, __qtablewidgetitem398);
        QTableWidgetItem *__qtablewidgetitem399 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(39, __qtablewidgetitem399);
        QTableWidgetItem *__qtablewidgetitem400 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(40, __qtablewidgetitem400);
        QTableWidgetItem *__qtablewidgetitem401 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(41, __qtablewidgetitem401);
        QTableWidgetItem *__qtablewidgetitem402 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(42, __qtablewidgetitem402);
        QTableWidgetItem *__qtablewidgetitem403 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(43, __qtablewidgetitem403);
        QTableWidgetItem *__qtablewidgetitem404 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(44, __qtablewidgetitem404);
        QTableWidgetItem *__qtablewidgetitem405 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(45, __qtablewidgetitem405);
        QTableWidgetItem *__qtablewidgetitem406 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(46, __qtablewidgetitem406);
        QTableWidgetItem *__qtablewidgetitem407 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(47, __qtablewidgetitem407);
        QTableWidgetItem *__qtablewidgetitem408 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(48, __qtablewidgetitem408);
        QTableWidgetItem *__qtablewidgetitem409 = new QTableWidgetItem();
        mediaTable->setVerticalHeaderItem(49, __qtablewidgetitem409);
        mediaTable->setObjectName(QString::fromUtf8("mediaTable"));
        sizePolicy2.setHeightForWidth(mediaTable->sizePolicy().hasHeightForWidth());
        mediaTable->setSizePolicy(sizePolicy2);
        mediaTable->setMinimumSize(QSize(345, 424));
        mediaTable->setMaximumSize(QSize(32767, 66666));
        mediaTable->setShowGrid(true);
        mediaTable->setSortingEnabled(true);
        mediaTable->setRowCount(50);
        mediaTable->setColumnCount(3);
        mediaTable->horizontalHeader()->setCascadingSectionResizes(false);
        mediaTable->horizontalHeader()->setProperty("showSortIndicator", QVariant(true));

        verticalLayout_36->addWidget(mediaTable);


        gridLayout_23->addWidget(mediaGroupBox, 0, 1, 2, 1);

        horizontalLayout_12 = new QHBoxLayout();
        horizontalLayout_12->setSpacing(6);
        horizontalLayout_12->setObjectName(QString::fromUtf8("horizontalLayout_12"));
        GroupBox53 = new QGroupBox(Gtab);
        GroupBox53->setObjectName(QString::fromUtf8("GroupBox53"));
        GroupBox53->setMaximumSize(QSize(270, 16777215));
        verticalLayout_33 = new QVBoxLayout(GroupBox53);
        verticalLayout_33->setSpacing(6);
        verticalLayout_33->setContentsMargins(11, 11, 11, 11);
        verticalLayout_33->setObjectName(QString::fromUtf8("verticalLayout_33"));
        geometryTable = new QTableWidget(GroupBox53);
        if (geometryTable->columnCount() < 2)
            geometryTable->setColumnCount(2);
        QTableWidgetItem *__qtablewidgetitem410 = new QTableWidgetItem();
        geometryTable->setHorizontalHeaderItem(0, __qtablewidgetitem410);
        QTableWidgetItem *__qtablewidgetitem411 = new QTableWidgetItem();
        geometryTable->setHorizontalHeaderItem(1, __qtablewidgetitem411);
        if (geometryTable->rowCount() < 50)
            geometryTable->setRowCount(50);
        QTableWidgetItem *__qtablewidgetitem412 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(0, __qtablewidgetitem412);
        QTableWidgetItem *__qtablewidgetitem413 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(1, __qtablewidgetitem413);
        QTableWidgetItem *__qtablewidgetitem414 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(2, __qtablewidgetitem414);
        QTableWidgetItem *__qtablewidgetitem415 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(3, __qtablewidgetitem415);
        QTableWidgetItem *__qtablewidgetitem416 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(4, __qtablewidgetitem416);
        QTableWidgetItem *__qtablewidgetitem417 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(5, __qtablewidgetitem417);
        QTableWidgetItem *__qtablewidgetitem418 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(6, __qtablewidgetitem418);
        QTableWidgetItem *__qtablewidgetitem419 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(7, __qtablewidgetitem419);
        QTableWidgetItem *__qtablewidgetitem420 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(8, __qtablewidgetitem420);
        QTableWidgetItem *__qtablewidgetitem421 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(9, __qtablewidgetitem421);
        QTableWidgetItem *__qtablewidgetitem422 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(10, __qtablewidgetitem422);
        QTableWidgetItem *__qtablewidgetitem423 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(11, __qtablewidgetitem423);
        QTableWidgetItem *__qtablewidgetitem424 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(12, __qtablewidgetitem424);
        QTableWidgetItem *__qtablewidgetitem425 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(13, __qtablewidgetitem425);
        QTableWidgetItem *__qtablewidgetitem426 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(14, __qtablewidgetitem426);
        QTableWidgetItem *__qtablewidgetitem427 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(15, __qtablewidgetitem427);
        QTableWidgetItem *__qtablewidgetitem428 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(16, __qtablewidgetitem428);
        QTableWidgetItem *__qtablewidgetitem429 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(17, __qtablewidgetitem429);
        QTableWidgetItem *__qtablewidgetitem430 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(18, __qtablewidgetitem430);
        QTableWidgetItem *__qtablewidgetitem431 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(19, __qtablewidgetitem431);
        QTableWidgetItem *__qtablewidgetitem432 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(20, __qtablewidgetitem432);
        QTableWidgetItem *__qtablewidgetitem433 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(21, __qtablewidgetitem433);
        QTableWidgetItem *__qtablewidgetitem434 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(22, __qtablewidgetitem434);
        QTableWidgetItem *__qtablewidgetitem435 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(23, __qtablewidgetitem435);
        QTableWidgetItem *__qtablewidgetitem436 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(24, __qtablewidgetitem436);
        QTableWidgetItem *__qtablewidgetitem437 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(25, __qtablewidgetitem437);
        QTableWidgetItem *__qtablewidgetitem438 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(26, __qtablewidgetitem438);
        QTableWidgetItem *__qtablewidgetitem439 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(27, __qtablewidgetitem439);
        QTableWidgetItem *__qtablewidgetitem440 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(28, __qtablewidgetitem440);
        QTableWidgetItem *__qtablewidgetitem441 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(29, __qtablewidgetitem441);
        QTableWidgetItem *__qtablewidgetitem442 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(30, __qtablewidgetitem442);
        QTableWidgetItem *__qtablewidgetitem443 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(31, __qtablewidgetitem443);
        QTableWidgetItem *__qtablewidgetitem444 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(32, __qtablewidgetitem444);
        QTableWidgetItem *__qtablewidgetitem445 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(33, __qtablewidgetitem445);
        QTableWidgetItem *__qtablewidgetitem446 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(34, __qtablewidgetitem446);
        QTableWidgetItem *__qtablewidgetitem447 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(35, __qtablewidgetitem447);
        QTableWidgetItem *__qtablewidgetitem448 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(36, __qtablewidgetitem448);
        QTableWidgetItem *__qtablewidgetitem449 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(37, __qtablewidgetitem449);
        QTableWidgetItem *__qtablewidgetitem450 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(38, __qtablewidgetitem450);
        QTableWidgetItem *__qtablewidgetitem451 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(39, __qtablewidgetitem451);
        QTableWidgetItem *__qtablewidgetitem452 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(40, __qtablewidgetitem452);
        QTableWidgetItem *__qtablewidgetitem453 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(41, __qtablewidgetitem453);
        QTableWidgetItem *__qtablewidgetitem454 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(42, __qtablewidgetitem454);
        QTableWidgetItem *__qtablewidgetitem455 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(43, __qtablewidgetitem455);
        QTableWidgetItem *__qtablewidgetitem456 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(44, __qtablewidgetitem456);
        QTableWidgetItem *__qtablewidgetitem457 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(45, __qtablewidgetitem457);
        QTableWidgetItem *__qtablewidgetitem458 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(46, __qtablewidgetitem458);
        QTableWidgetItem *__qtablewidgetitem459 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(47, __qtablewidgetitem459);
        QTableWidgetItem *__qtablewidgetitem460 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(48, __qtablewidgetitem460);
        QTableWidgetItem *__qtablewidgetitem461 = new QTableWidgetItem();
        geometryTable->setVerticalHeaderItem(49, __qtablewidgetitem461);
        geometryTable->setObjectName(QString::fromUtf8("geometryTable"));
        sizePolicy2.setHeightForWidth(geometryTable->sizePolicy().hasHeightForWidth());
        geometryTable->setSizePolicy(sizePolicy2);
        geometryTable->setMinimumSize(QSize(235, 350));
        geometryTable->setMaximumSize(QSize(6666, 6666));
        geometryTable->setFocusPolicy(Qt::WheelFocus);
        geometryTable->setRowCount(50);
        geometryTable->setColumnCount(2);

        verticalLayout_33->addWidget(geometryTable);


        horizontalLayout_12->addWidget(GroupBox53);

        GroupBox54 = new QGroupBox(Gtab);
        GroupBox54->setObjectName(QString::fromUtf8("GroupBox54"));
        GroupBox54->setMaximumSize(QSize(181, 16777215));
        verticalLayout_34 = new QVBoxLayout(GroupBox54);
        verticalLayout_34->setSpacing(6);
        verticalLayout_34->setContentsMargins(11, 11, 11, 11);
        verticalLayout_34->setObjectName(QString::fromUtf8("verticalLayout_34"));
        cylTable = new QTableWidget(GroupBox54);
        if (cylTable->columnCount() < 1)
            cylTable->setColumnCount(1);
        QTableWidgetItem *__qtablewidgetitem462 = new QTableWidgetItem();
        cylTable->setHorizontalHeaderItem(0, __qtablewidgetitem462);
        if (cylTable->rowCount() < 50)
            cylTable->setRowCount(50);
        QTableWidgetItem *__qtablewidgetitem463 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(0, __qtablewidgetitem463);
        QTableWidgetItem *__qtablewidgetitem464 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(1, __qtablewidgetitem464);
        QTableWidgetItem *__qtablewidgetitem465 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(2, __qtablewidgetitem465);
        QTableWidgetItem *__qtablewidgetitem466 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(3, __qtablewidgetitem466);
        QTableWidgetItem *__qtablewidgetitem467 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(4, __qtablewidgetitem467);
        QTableWidgetItem *__qtablewidgetitem468 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(5, __qtablewidgetitem468);
        QTableWidgetItem *__qtablewidgetitem469 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(6, __qtablewidgetitem469);
        QTableWidgetItem *__qtablewidgetitem470 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(7, __qtablewidgetitem470);
        QTableWidgetItem *__qtablewidgetitem471 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(8, __qtablewidgetitem471);
        QTableWidgetItem *__qtablewidgetitem472 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(9, __qtablewidgetitem472);
        QTableWidgetItem *__qtablewidgetitem473 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(10, __qtablewidgetitem473);
        QTableWidgetItem *__qtablewidgetitem474 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(11, __qtablewidgetitem474);
        QTableWidgetItem *__qtablewidgetitem475 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(12, __qtablewidgetitem475);
        QTableWidgetItem *__qtablewidgetitem476 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(13, __qtablewidgetitem476);
        QTableWidgetItem *__qtablewidgetitem477 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(14, __qtablewidgetitem477);
        QTableWidgetItem *__qtablewidgetitem478 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(15, __qtablewidgetitem478);
        QTableWidgetItem *__qtablewidgetitem479 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(16, __qtablewidgetitem479);
        QTableWidgetItem *__qtablewidgetitem480 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(17, __qtablewidgetitem480);
        QTableWidgetItem *__qtablewidgetitem481 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(18, __qtablewidgetitem481);
        QTableWidgetItem *__qtablewidgetitem482 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(19, __qtablewidgetitem482);
        QTableWidgetItem *__qtablewidgetitem483 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(20, __qtablewidgetitem483);
        QTableWidgetItem *__qtablewidgetitem484 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(21, __qtablewidgetitem484);
        QTableWidgetItem *__qtablewidgetitem485 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(22, __qtablewidgetitem485);
        QTableWidgetItem *__qtablewidgetitem486 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(23, __qtablewidgetitem486);
        QTableWidgetItem *__qtablewidgetitem487 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(24, __qtablewidgetitem487);
        QTableWidgetItem *__qtablewidgetitem488 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(25, __qtablewidgetitem488);
        QTableWidgetItem *__qtablewidgetitem489 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(26, __qtablewidgetitem489);
        QTableWidgetItem *__qtablewidgetitem490 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(27, __qtablewidgetitem490);
        QTableWidgetItem *__qtablewidgetitem491 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(28, __qtablewidgetitem491);
        QTableWidgetItem *__qtablewidgetitem492 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(29, __qtablewidgetitem492);
        QTableWidgetItem *__qtablewidgetitem493 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(30, __qtablewidgetitem493);
        QTableWidgetItem *__qtablewidgetitem494 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(31, __qtablewidgetitem494);
        QTableWidgetItem *__qtablewidgetitem495 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(32, __qtablewidgetitem495);
        QTableWidgetItem *__qtablewidgetitem496 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(33, __qtablewidgetitem496);
        QTableWidgetItem *__qtablewidgetitem497 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(34, __qtablewidgetitem497);
        QTableWidgetItem *__qtablewidgetitem498 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(35, __qtablewidgetitem498);
        QTableWidgetItem *__qtablewidgetitem499 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(36, __qtablewidgetitem499);
        QTableWidgetItem *__qtablewidgetitem500 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(37, __qtablewidgetitem500);
        QTableWidgetItem *__qtablewidgetitem501 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(38, __qtablewidgetitem501);
        QTableWidgetItem *__qtablewidgetitem502 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(39, __qtablewidgetitem502);
        QTableWidgetItem *__qtablewidgetitem503 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(40, __qtablewidgetitem503);
        QTableWidgetItem *__qtablewidgetitem504 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(41, __qtablewidgetitem504);
        QTableWidgetItem *__qtablewidgetitem505 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(42, __qtablewidgetitem505);
        QTableWidgetItem *__qtablewidgetitem506 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(43, __qtablewidgetitem506);
        QTableWidgetItem *__qtablewidgetitem507 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(44, __qtablewidgetitem507);
        QTableWidgetItem *__qtablewidgetitem508 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(45, __qtablewidgetitem508);
        QTableWidgetItem *__qtablewidgetitem509 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(46, __qtablewidgetitem509);
        QTableWidgetItem *__qtablewidgetitem510 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(47, __qtablewidgetitem510);
        QTableWidgetItem *__qtablewidgetitem511 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(48, __qtablewidgetitem511);
        QTableWidgetItem *__qtablewidgetitem512 = new QTableWidgetItem();
        cylTable->setVerticalHeaderItem(49, __qtablewidgetitem512);
        cylTable->setObjectName(QString::fromUtf8("cylTable"));
        sizePolicy2.setHeightForWidth(cylTable->sizePolicy().hasHeightForWidth());
        cylTable->setSizePolicy(sizePolicy2);
        cylTable->setMinimumSize(QSize(172, 350));
        cylTable->setMaximumSize(QSize(66666, 66666));
        cylTable->setMouseTracking(false);
        cylTable->setRowCount(50);
        cylTable->setColumnCount(1);

        verticalLayout_34->addWidget(cylTable);


        horizontalLayout_12->addWidget(GroupBox54);


        gridLayout_23->addLayout(horizontalLayout_12, 1, 0, 1, 1);


        gridLayout_24->addLayout(gridLayout_23, 0, 0, 1, 1);

        TabWidgetRZ->addTab(Gtab, QString());
        CItab = new QWidget();
        CItab->setObjectName(QString::fromUtf8("CItab"));
        layoutWidget = new QWidget(CItab);
        layoutWidget->setObjectName(QString::fromUtf8("layoutWidget"));
        layoutWidget->setGeometry(QRect(10, 30, 901, 451));
        horizontalLayout_6 = new QHBoxLayout(layoutWidget);
        horizontalLayout_6->setSpacing(6);
        horizontalLayout_6->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        horizontalLayout_6->setContentsMargins(0, 0, 0, 0);
        verticalLayout_7 = new QVBoxLayout();
        verticalLayout_7->setSpacing(6);
        verticalLayout_7->setObjectName(QString::fromUtf8("verticalLayout_7"));
        cavityGroupBox = new QGroupBox(layoutWidget);
        cavityGroupBox->setObjectName(QString::fromUtf8("cavityGroupBox"));
        horizontalLayout_4 = new QHBoxLayout(cavityGroupBox);
        horizontalLayout_4->setSpacing(6);
        horizontalLayout_4->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setSpacing(6);
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        verticalLayout_6 = new QVBoxLayout();
        verticalLayout_6->setSpacing(6);
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_6->addItem(verticalSpacer);

        ButtonGroup25 = new QGroupBox(cavityGroupBox);
        ButtonGroup25->setObjectName(QString::fromUtf8("ButtonGroup25"));
        QSizePolicy sizePolicy12(QSizePolicy::Maximum, QSizePolicy::Preferred);
        sizePolicy12.setHorizontalStretch(0);
        sizePolicy12.setVerticalStretch(0);
        sizePolicy12.setHeightForWidth(ButtonGroup25->sizePolicy().hasHeightForWidth());
        ButtonGroup25->setSizePolicy(sizePolicy12);
        verticalLayout_4 = new QVBoxLayout(ButtonGroup25);
        verticalLayout_4->setSpacing(6);
        verticalLayout_4->setContentsMargins(11, 11, 11, 11);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        thimbleRadioButton = new QRadioButton(ButtonGroup25);
        thimbleRadioButton->setObjectName(QString::fromUtf8("thimbleRadioButton"));
        thimbleRadioButton->setChecked(true);

        verticalLayout_4->addWidget(thimbleRadioButton);

        parallelplateRadioButton = new QRadioButton(ButtonGroup25);
        parallelplateRadioButton->setObjectName(QString::fromUtf8("parallelplateRadioButton"));
        parallelplateRadioButton->setChecked(false);

        verticalLayout_4->addWidget(parallelplateRadioButton);


        verticalLayout_6->addWidget(ButtonGroup25);

        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_6->addItem(verticalSpacer_2);


        horizontalLayout_3->addLayout(verticalLayout_6);

        verticalLayout_5 = new QVBoxLayout();
        verticalLayout_5->setSpacing(6);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        hboxLayout27 = new QHBoxLayout();
        hboxLayout27->setSpacing(6);
        hboxLayout27->setObjectName(QString::fromUtf8("hboxLayout27"));
        wallLabel = new QLabel(cavityGroupBox);
        wallLabel->setObjectName(QString::fromUtf8("wallLabel"));
        wallLabel->setWordWrap(false);

        hboxLayout27->addWidget(wallLabel);

        wallthickEdit = new QLineEdit(cavityGroupBox);
        wallthickEdit->setObjectName(QString::fromUtf8("wallthickEdit"));

        hboxLayout27->addWidget(wallthickEdit);


        verticalLayout_5->addLayout(hboxLayout27);

        hboxLayout28 = new QHBoxLayout();
        hboxLayout28->setSpacing(6);
        hboxLayout28->setObjectName(QString::fromUtf8("hboxLayout28"));
        cavradLabel = new QLabel(cavityGroupBox);
        cavradLabel->setObjectName(QString::fromUtf8("cavradLabel"));
        cavradLabel->setWordWrap(false);

        hboxLayout28->addWidget(cavradLabel);

        cavradEdit = new QLineEdit(cavityGroupBox);
        cavradEdit->setObjectName(QString::fromUtf8("cavradEdit"));

        hboxLayout28->addWidget(cavradEdit);


        verticalLayout_5->addLayout(hboxLayout28);

        hboxLayout29 = new QHBoxLayout();
        hboxLayout29->setSpacing(6);
        hboxLayout29->setObjectName(QString::fromUtf8("hboxLayout29"));
        cavlenLabel = new QLabel(cavityGroupBox);
        cavlenLabel->setObjectName(QString::fromUtf8("cavlenLabel"));
        cavlenLabel->setWordWrap(false);

        hboxLayout29->addWidget(cavlenLabel);

        cavlenEdit = new QLineEdit(cavityGroupBox);
        cavlenEdit->setObjectName(QString::fromUtf8("cavlenEdit"));

        hboxLayout29->addWidget(cavlenEdit);


        verticalLayout_5->addLayout(hboxLayout29);

        hboxLayout30 = new QHBoxLayout();
        hboxLayout30->setSpacing(6);
        hboxLayout30->setObjectName(QString::fromUtf8("hboxLayout30"));
        electradLabel = new QLabel(cavityGroupBox);
        electradLabel->setObjectName(QString::fromUtf8("electradLabel"));
        electradLabel->setWordWrap(false);

        hboxLayout30->addWidget(electradLabel);

        electradEdit = new QLineEdit(cavityGroupBox);
        electradEdit->setObjectName(QString::fromUtf8("electradEdit"));

        hboxLayout30->addWidget(electradEdit);


        verticalLayout_5->addLayout(hboxLayout30);

        hboxLayout31 = new QHBoxLayout();
        hboxLayout31->setSpacing(6);
        hboxLayout31->setObjectName(QString::fromUtf8("hboxLayout31"));
        wallmaterialLabel = new QLabel(cavityGroupBox);
        wallmaterialLabel->setObjectName(QString::fromUtf8("wallmaterialLabel"));
        wallmaterialLabel->setWordWrap(false);

        hboxLayout31->addWidget(wallmaterialLabel);

        wallmaterialComboBox = new QComboBox(cavityGroupBox);
        wallmaterialComboBox->setObjectName(QString::fromUtf8("wallmaterialComboBox"));
        sizePolicy4.setHeightForWidth(wallmaterialComboBox->sizePolicy().hasHeightForWidth());
        wallmaterialComboBox->setSizePolicy(sizePolicy4);
        wallmaterialComboBox->setEditable(true);
        wallmaterialComboBox->setInsertPolicy(QComboBox::InsertAtTop);
        wallmaterialComboBox->setAutoCompletion(true);

        hboxLayout31->addWidget(wallmaterialComboBox);


        verticalLayout_5->addLayout(hboxLayout31);

        hboxLayout32 = new QHBoxLayout();
        hboxLayout32->setSpacing(6);
        hboxLayout32->setObjectName(QString::fromUtf8("hboxLayout32"));
        electrmatLabel = new QLabel(cavityGroupBox);
        electrmatLabel->setObjectName(QString::fromUtf8("electrmatLabel"));
        electrmatLabel->setWordWrap(false);

        hboxLayout32->addWidget(electrmatLabel);

        electrmatComboBox = new QComboBox(cavityGroupBox);
        electrmatComboBox->setObjectName(QString::fromUtf8("electrmatComboBox"));
        sizePolicy4.setHeightForWidth(electrmatComboBox->sizePolicy().hasHeightForWidth());
        electrmatComboBox->setSizePolicy(sizePolicy4);
        electrmatComboBox->setEditable(true);
        electrmatComboBox->setInsertPolicy(QComboBox::InsertAtTop);
        electrmatComboBox->setAutoCompletion(true);

        hboxLayout32->addWidget(electrmatComboBox);


        verticalLayout_5->addLayout(hboxLayout32);


        horizontalLayout_3->addLayout(verticalLayout_5);


        horizontalLayout_4->addLayout(horizontalLayout_3);


        verticalLayout_7->addWidget(cavityGroupBox);

        verticalSpacer_3 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_7->addItem(verticalSpacer_3);


        horizontalLayout_6->addLayout(verticalLayout_7);

        gr_indGroupBox = new QGroupBox(layoutWidget);
        gr_indGroupBox->setObjectName(QString::fromUtf8("gr_indGroupBox"));
        horizontalLayout_5 = new QHBoxLayout(gr_indGroupBox);
        horizontalLayout_5->setSpacing(6);
        horizontalLayout_5->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        GroupBox56 = new QGroupBox(gr_indGroupBox);
        GroupBox56->setObjectName(QString::fromUtf8("GroupBox56"));
        cavTable = new QTableWidget(GroupBox56);
        if (cavTable->columnCount() < 1)
            cavTable->setColumnCount(1);
        QTableWidgetItem *__qtablewidgetitem513 = new QTableWidgetItem();
        cavTable->setHorizontalHeaderItem(0, __qtablewidgetitem513);
        if (cavTable->rowCount() < 50)
            cavTable->setRowCount(50);
        QTableWidgetItem *__qtablewidgetitem514 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(0, __qtablewidgetitem514);
        QTableWidgetItem *__qtablewidgetitem515 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(1, __qtablewidgetitem515);
        QTableWidgetItem *__qtablewidgetitem516 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(2, __qtablewidgetitem516);
        QTableWidgetItem *__qtablewidgetitem517 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(3, __qtablewidgetitem517);
        QTableWidgetItem *__qtablewidgetitem518 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(4, __qtablewidgetitem518);
        QTableWidgetItem *__qtablewidgetitem519 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(5, __qtablewidgetitem519);
        QTableWidgetItem *__qtablewidgetitem520 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(6, __qtablewidgetitem520);
        QTableWidgetItem *__qtablewidgetitem521 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(7, __qtablewidgetitem521);
        QTableWidgetItem *__qtablewidgetitem522 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(8, __qtablewidgetitem522);
        QTableWidgetItem *__qtablewidgetitem523 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(9, __qtablewidgetitem523);
        QTableWidgetItem *__qtablewidgetitem524 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(10, __qtablewidgetitem524);
        QTableWidgetItem *__qtablewidgetitem525 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(11, __qtablewidgetitem525);
        QTableWidgetItem *__qtablewidgetitem526 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(12, __qtablewidgetitem526);
        QTableWidgetItem *__qtablewidgetitem527 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(13, __qtablewidgetitem527);
        QTableWidgetItem *__qtablewidgetitem528 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(14, __qtablewidgetitem528);
        QTableWidgetItem *__qtablewidgetitem529 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(15, __qtablewidgetitem529);
        QTableWidgetItem *__qtablewidgetitem530 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(16, __qtablewidgetitem530);
        QTableWidgetItem *__qtablewidgetitem531 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(17, __qtablewidgetitem531);
        QTableWidgetItem *__qtablewidgetitem532 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(18, __qtablewidgetitem532);
        QTableWidgetItem *__qtablewidgetitem533 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(19, __qtablewidgetitem533);
        QTableWidgetItem *__qtablewidgetitem534 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(20, __qtablewidgetitem534);
        QTableWidgetItem *__qtablewidgetitem535 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(21, __qtablewidgetitem535);
        QTableWidgetItem *__qtablewidgetitem536 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(22, __qtablewidgetitem536);
        QTableWidgetItem *__qtablewidgetitem537 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(23, __qtablewidgetitem537);
        QTableWidgetItem *__qtablewidgetitem538 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(24, __qtablewidgetitem538);
        QTableWidgetItem *__qtablewidgetitem539 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(25, __qtablewidgetitem539);
        QTableWidgetItem *__qtablewidgetitem540 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(26, __qtablewidgetitem540);
        QTableWidgetItem *__qtablewidgetitem541 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(27, __qtablewidgetitem541);
        QTableWidgetItem *__qtablewidgetitem542 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(28, __qtablewidgetitem542);
        QTableWidgetItem *__qtablewidgetitem543 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(29, __qtablewidgetitem543);
        QTableWidgetItem *__qtablewidgetitem544 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(30, __qtablewidgetitem544);
        QTableWidgetItem *__qtablewidgetitem545 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(31, __qtablewidgetitem545);
        QTableWidgetItem *__qtablewidgetitem546 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(32, __qtablewidgetitem546);
        QTableWidgetItem *__qtablewidgetitem547 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(33, __qtablewidgetitem547);
        QTableWidgetItem *__qtablewidgetitem548 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(34, __qtablewidgetitem548);
        QTableWidgetItem *__qtablewidgetitem549 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(35, __qtablewidgetitem549);
        QTableWidgetItem *__qtablewidgetitem550 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(36, __qtablewidgetitem550);
        QTableWidgetItem *__qtablewidgetitem551 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(37, __qtablewidgetitem551);
        QTableWidgetItem *__qtablewidgetitem552 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(38, __qtablewidgetitem552);
        QTableWidgetItem *__qtablewidgetitem553 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(39, __qtablewidgetitem553);
        QTableWidgetItem *__qtablewidgetitem554 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(40, __qtablewidgetitem554);
        QTableWidgetItem *__qtablewidgetitem555 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(41, __qtablewidgetitem555);
        QTableWidgetItem *__qtablewidgetitem556 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(42, __qtablewidgetitem556);
        QTableWidgetItem *__qtablewidgetitem557 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(43, __qtablewidgetitem557);
        QTableWidgetItem *__qtablewidgetitem558 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(44, __qtablewidgetitem558);
        QTableWidgetItem *__qtablewidgetitem559 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(45, __qtablewidgetitem559);
        QTableWidgetItem *__qtablewidgetitem560 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(46, __qtablewidgetitem560);
        QTableWidgetItem *__qtablewidgetitem561 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(47, __qtablewidgetitem561);
        QTableWidgetItem *__qtablewidgetitem562 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(48, __qtablewidgetitem562);
        QTableWidgetItem *__qtablewidgetitem563 = new QTableWidgetItem();
        cavTable->setVerticalHeaderItem(49, __qtablewidgetitem563);
        cavTable->setObjectName(QString::fromUtf8("cavTable"));
        cavTable->setGeometry(QRect(19, 29, 150, 250));
        cavTable->setRowCount(50);
        cavTable->setColumnCount(1);

        horizontalLayout_5->addWidget(GroupBox56);


        horizontalLayout_6->addWidget(gr_indGroupBox);

        TabWidgetRZ->addTab(CItab, QString());
        MDTab = new QWidget();
        MDTab->setObjectName(QString::fromUtf8("MDTab"));
        verticalLayout_52 = new QVBoxLayout(MDTab);
        verticalLayout_52->setSpacing(6);
        verticalLayout_52->setContentsMargins(11, 11, 11, 11);
        verticalLayout_52->setObjectName(QString::fromUtf8("verticalLayout_52"));
        verticalLayout_51 = new QVBoxLayout();
        verticalLayout_51->setSpacing(6);
        verticalLayout_51->setObjectName(QString::fromUtf8("verticalLayout_51"));
        horizontalLayout_28 = new QHBoxLayout();
        horizontalLayout_28->setSpacing(6);
        horizontalLayout_28->setObjectName(QString::fromUtf8("horizontalLayout_28"));
        ELGroupBox = new QGroupBox(MDTab);
        ELGroupBox->setObjectName(QString::fromUtf8("ELGroupBox"));
        verticalLayout_47 = new QVBoxLayout(ELGroupBox);
        verticalLayout_47->setSpacing(6);
        verticalLayout_47->setContentsMargins(11, 11, 11, 11);
        verticalLayout_47->setObjectName(QString::fromUtf8("verticalLayout_47"));
        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setSpacing(6);
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));
        vboxLayout26 = new QVBoxLayout();
        vboxLayout26->setSpacing(6);
        vboxLayout26->setObjectName(QString::fromUtf8("vboxLayout26"));
        hboxLayout33 = new QHBoxLayout();
        hboxLayout33->setSpacing(6);
        hboxLayout33->setObjectName(QString::fromUtf8("hboxLayout33"));
        AELabel = new QLabel(ELGroupBox);
        AELabel->setObjectName(QString::fromUtf8("AELabel"));
        AELabel->setWordWrap(false);

        hboxLayout33->addWidget(AELabel);

        AEEdit = new QLineEdit(ELGroupBox);
        AEEdit->setObjectName(QString::fromUtf8("AEEdit"));

        hboxLayout33->addWidget(AEEdit);


        vboxLayout26->addLayout(hboxLayout33);

        hboxLayout34 = new QHBoxLayout();
        hboxLayout34->setSpacing(6);
        hboxLayout34->setObjectName(QString::fromUtf8("hboxLayout34"));
        APLabel = new QLabel(ELGroupBox);
        APLabel->setObjectName(QString::fromUtf8("APLabel"));
        APLabel->setWordWrap(false);

        hboxLayout34->addWidget(APLabel);

        APEdit = new QLineEdit(ELGroupBox);
        APEdit->setObjectName(QString::fromUtf8("APEdit"));

        hboxLayout34->addWidget(APEdit);


        vboxLayout26->addLayout(hboxLayout34);


        horizontalLayout_7->addLayout(vboxLayout26);

        vboxLayout27 = new QVBoxLayout();
        vboxLayout27->setSpacing(6);
        vboxLayout27->setObjectName(QString::fromUtf8("vboxLayout27"));
        hboxLayout35 = new QHBoxLayout();
        hboxLayout35->setSpacing(6);
        hboxLayout35->setObjectName(QString::fromUtf8("hboxLayout35"));
        UELabel = new QLabel(ELGroupBox);
        UELabel->setObjectName(QString::fromUtf8("UELabel"));
        UELabel->setWordWrap(false);

        hboxLayout35->addWidget(UELabel);

        UEEdit = new QLineEdit(ELGroupBox);
        UEEdit->setObjectName(QString::fromUtf8("UEEdit"));

        hboxLayout35->addWidget(UEEdit);


        vboxLayout27->addLayout(hboxLayout35);

        hboxLayout36 = new QHBoxLayout();
        hboxLayout36->setSpacing(6);
        hboxLayout36->setObjectName(QString::fromUtf8("hboxLayout36"));
        UPLabel = new QLabel(ELGroupBox);
        UPLabel->setObjectName(QString::fromUtf8("UPLabel"));
        UPLabel->setWordWrap(false);

        hboxLayout36->addWidget(UPLabel);

        UPEdit = new QLineEdit(ELGroupBox);
        UPEdit->setObjectName(QString::fromUtf8("UPEdit"));

        hboxLayout36->addWidget(UPEdit);


        vboxLayout27->addLayout(hboxLayout36);


        horizontalLayout_7->addLayout(vboxLayout27);


        verticalLayout_47->addLayout(horizontalLayout_7);


        horizontalLayout_28->addWidget(ELGroupBox);

        MDGroupBox = new QGroupBox(MDTab);
        MDGroupBox->setObjectName(QString::fromUtf8("MDGroupBox"));
        verticalLayout_48 = new QVBoxLayout(MDGroupBox);
        verticalLayout_48->setSpacing(6);
        verticalLayout_48->setContentsMargins(11, 11, 11, 11);
        verticalLayout_48->setObjectName(QString::fromUtf8("verticalLayout_48"));
        hboxLayout37 = new QHBoxLayout();
        hboxLayout37->setSpacing(6);
        hboxLayout37->setObjectName(QString::fromUtf8("hboxLayout37"));
        MDFEdit = new QLineEdit(MDGroupBox);
        MDFEdit->setObjectName(QString::fromUtf8("MDFEdit"));

        hboxLayout37->addWidget(MDFEdit);

        MDFileButton = new QPushButton(MDGroupBox);
        MDFileButton->setObjectName(QString::fromUtf8("MDFileButton"));
        MDFileButton->setAutoDefault(false);

        hboxLayout37->addWidget(MDFileButton);


        verticalLayout_48->addLayout(hboxLayout37);


        horizontalLayout_28->addWidget(MDGroupBox);


        verticalLayout_51->addLayout(horizontalLayout_28);

        inpmediaGroupBox = new QGroupBox(MDTab);
        inpmediaGroupBox->setObjectName(QString::fromUtf8("inpmediaGroupBox"));
        verticalLayout_50 = new QVBoxLayout(inpmediaGroupBox);
        verticalLayout_50->setSpacing(6);
        verticalLayout_50->setContentsMargins(11, 11, 11, 11);
        verticalLayout_50->setObjectName(QString::fromUtf8("verticalLayout_50"));
        verticalLayout_49 = new QVBoxLayout();
        verticalLayout_49->setSpacing(6);
        verticalLayout_49->setObjectName(QString::fromUtf8("verticalLayout_49"));
        horizontalLayout_24 = new QHBoxLayout();
        horizontalLayout_24->setSpacing(6);
        horizontalLayout_24->setObjectName(QString::fromUtf8("horizontalLayout_24"));
        compositionGroupBox = new QGroupBox(inpmediaGroupBox);
        compositionGroupBox->setObjectName(QString::fromUtf8("compositionGroupBox"));
        verticalLayout_8 = new QVBoxLayout(compositionGroupBox);
        verticalLayout_8->setSpacing(6);
        verticalLayout_8->setContentsMargins(11, 11, 11, 11);
        verticalLayout_8->setObjectName(QString::fromUtf8("verticalLayout_8"));
        pz_or_rhozTable = new QTableWidget(compositionGroupBox);
        if (pz_or_rhozTable->columnCount() < 2)
            pz_or_rhozTable->setColumnCount(2);
        QTableWidgetItem *__qtablewidgetitem564 = new QTableWidgetItem();
        pz_or_rhozTable->setHorizontalHeaderItem(0, __qtablewidgetitem564);
        QTableWidgetItem *__qtablewidgetitem565 = new QTableWidgetItem();
        pz_or_rhozTable->setHorizontalHeaderItem(1, __qtablewidgetitem565);
        if (pz_or_rhozTable->rowCount() < 50)
            pz_or_rhozTable->setRowCount(50);
        QTableWidgetItem *__qtablewidgetitem566 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(0, __qtablewidgetitem566);
        QTableWidgetItem *__qtablewidgetitem567 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(1, __qtablewidgetitem567);
        QTableWidgetItem *__qtablewidgetitem568 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(2, __qtablewidgetitem568);
        QTableWidgetItem *__qtablewidgetitem569 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(3, __qtablewidgetitem569);
        QTableWidgetItem *__qtablewidgetitem570 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(4, __qtablewidgetitem570);
        QTableWidgetItem *__qtablewidgetitem571 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(5, __qtablewidgetitem571);
        QTableWidgetItem *__qtablewidgetitem572 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(6, __qtablewidgetitem572);
        QTableWidgetItem *__qtablewidgetitem573 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(7, __qtablewidgetitem573);
        QTableWidgetItem *__qtablewidgetitem574 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(8, __qtablewidgetitem574);
        QTableWidgetItem *__qtablewidgetitem575 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(9, __qtablewidgetitem575);
        QTableWidgetItem *__qtablewidgetitem576 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(10, __qtablewidgetitem576);
        QTableWidgetItem *__qtablewidgetitem577 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(11, __qtablewidgetitem577);
        QTableWidgetItem *__qtablewidgetitem578 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(12, __qtablewidgetitem578);
        QTableWidgetItem *__qtablewidgetitem579 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(13, __qtablewidgetitem579);
        QTableWidgetItem *__qtablewidgetitem580 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(14, __qtablewidgetitem580);
        QTableWidgetItem *__qtablewidgetitem581 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(15, __qtablewidgetitem581);
        QTableWidgetItem *__qtablewidgetitem582 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(16, __qtablewidgetitem582);
        QTableWidgetItem *__qtablewidgetitem583 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(17, __qtablewidgetitem583);
        QTableWidgetItem *__qtablewidgetitem584 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(18, __qtablewidgetitem584);
        QTableWidgetItem *__qtablewidgetitem585 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(19, __qtablewidgetitem585);
        QTableWidgetItem *__qtablewidgetitem586 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(20, __qtablewidgetitem586);
        QTableWidgetItem *__qtablewidgetitem587 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(21, __qtablewidgetitem587);
        QTableWidgetItem *__qtablewidgetitem588 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(22, __qtablewidgetitem588);
        QTableWidgetItem *__qtablewidgetitem589 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(23, __qtablewidgetitem589);
        QTableWidgetItem *__qtablewidgetitem590 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(24, __qtablewidgetitem590);
        QTableWidgetItem *__qtablewidgetitem591 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(25, __qtablewidgetitem591);
        QTableWidgetItem *__qtablewidgetitem592 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(26, __qtablewidgetitem592);
        QTableWidgetItem *__qtablewidgetitem593 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(27, __qtablewidgetitem593);
        QTableWidgetItem *__qtablewidgetitem594 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(28, __qtablewidgetitem594);
        QTableWidgetItem *__qtablewidgetitem595 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(29, __qtablewidgetitem595);
        QTableWidgetItem *__qtablewidgetitem596 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(30, __qtablewidgetitem596);
        QTableWidgetItem *__qtablewidgetitem597 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(31, __qtablewidgetitem597);
        QTableWidgetItem *__qtablewidgetitem598 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(32, __qtablewidgetitem598);
        QTableWidgetItem *__qtablewidgetitem599 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(33, __qtablewidgetitem599);
        QTableWidgetItem *__qtablewidgetitem600 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(34, __qtablewidgetitem600);
        QTableWidgetItem *__qtablewidgetitem601 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(35, __qtablewidgetitem601);
        QTableWidgetItem *__qtablewidgetitem602 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(36, __qtablewidgetitem602);
        QTableWidgetItem *__qtablewidgetitem603 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(37, __qtablewidgetitem603);
        QTableWidgetItem *__qtablewidgetitem604 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(38, __qtablewidgetitem604);
        QTableWidgetItem *__qtablewidgetitem605 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(39, __qtablewidgetitem605);
        QTableWidgetItem *__qtablewidgetitem606 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(40, __qtablewidgetitem606);
        QTableWidgetItem *__qtablewidgetitem607 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(41, __qtablewidgetitem607);
        QTableWidgetItem *__qtablewidgetitem608 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(42, __qtablewidgetitem608);
        QTableWidgetItem *__qtablewidgetitem609 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(43, __qtablewidgetitem609);
        QTableWidgetItem *__qtablewidgetitem610 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(44, __qtablewidgetitem610);
        QTableWidgetItem *__qtablewidgetitem611 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(45, __qtablewidgetitem611);
        QTableWidgetItem *__qtablewidgetitem612 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(46, __qtablewidgetitem612);
        QTableWidgetItem *__qtablewidgetitem613 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(47, __qtablewidgetitem613);
        QTableWidgetItem *__qtablewidgetitem614 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(48, __qtablewidgetitem614);
        QTableWidgetItem *__qtablewidgetitem615 = new QTableWidgetItem();
        pz_or_rhozTable->setVerticalHeaderItem(49, __qtablewidgetitem615);
        pz_or_rhozTable->setObjectName(QString::fromUtf8("pz_or_rhozTable"));
        pz_or_rhozTable->setRowCount(50);
        pz_or_rhozTable->setColumnCount(2);

        verticalLayout_8->addWidget(pz_or_rhozTable);


        horizontalLayout_24->addWidget(compositionGroupBox);

        verticalLayout_46 = new QVBoxLayout();
        verticalLayout_46->setSpacing(6);
        verticalLayout_46->setObjectName(QString::fromUtf8("verticalLayout_46"));
        medNameGroupBox = new QGroupBox(inpmediaGroupBox);
        medNameGroupBox->setObjectName(QString::fromUtf8("medNameGroupBox"));
        verticalLayout_9 = new QVBoxLayout(medNameGroupBox);
        verticalLayout_9->setSpacing(6);
        verticalLayout_9->setContentsMargins(11, 11, 11, 11);
        verticalLayout_9->setObjectName(QString::fromUtf8("verticalLayout_9"));
        inpmediumComboBox = new QComboBox(medNameGroupBox);
        inpmediumComboBox->setObjectName(QString::fromUtf8("inpmediumComboBox"));
        QSizePolicy sizePolicy13(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);
        sizePolicy13.setHorizontalStretch(0);
        sizePolicy13.setVerticalStretch(0);
        sizePolicy13.setHeightForWidth(inpmediumComboBox->sizePolicy().hasHeightForWidth());
        inpmediumComboBox->setSizePolicy(sizePolicy13);
        inpmediumComboBox->setEditable(true);
        inpmediumComboBox->setInsertPolicy(QComboBox::InsertAtCurrent);

        verticalLayout_9->addWidget(inpmediumComboBox);


        verticalLayout_46->addWidget(medNameGroupBox);

        horizontalLayout_19 = new QHBoxLayout();
        horizontalLayout_19->setSpacing(6);
        horizontalLayout_19->setObjectName(QString::fromUtf8("horizontalLayout_19"));
        medTypeGroupBox = new QGroupBox(inpmediaGroupBox);
        medTypeGroupBox->setObjectName(QString::fromUtf8("medTypeGroupBox"));
        verticalLayout_37 = new QVBoxLayout(medTypeGroupBox);
        verticalLayout_37->setSpacing(6);
        verticalLayout_37->setContentsMargins(11, 11, 11, 11);
        verticalLayout_37->setObjectName(QString::fromUtf8("verticalLayout_37"));
        medTypeComboBox = new QComboBox(medTypeGroupBox);
        medTypeComboBox->addItem(QString());
        medTypeComboBox->addItem(QString());
        medTypeComboBox->addItem(QString());
        medTypeComboBox->setObjectName(QString::fromUtf8("medTypeComboBox"));
        medTypeComboBox->setEditable(false);
        medTypeComboBox->setMaxVisibleItems(3);

        verticalLayout_37->addWidget(medTypeComboBox);


        horizontalLayout_19->addWidget(medTypeGroupBox);

        rhoGroupBox = new QGroupBox(inpmediaGroupBox);
        rhoGroupBox->setObjectName(QString::fromUtf8("rhoGroupBox"));
        verticalLayout_44 = new QVBoxLayout(rhoGroupBox);
        verticalLayout_44->setSpacing(6);
        verticalLayout_44->setContentsMargins(11, 11, 11, 11);
        verticalLayout_44->setObjectName(QString::fromUtf8("verticalLayout_44"));
        horizontalLayout_8 = new QHBoxLayout();
        horizontalLayout_8->setSpacing(6);
        horizontalLayout_8->setObjectName(QString::fromUtf8("horizontalLayout_8"));
        rhoEdit = new QLineEdit(rhoGroupBox);
        rhoEdit->setObjectName(QString::fromUtf8("rhoEdit"));

        horizontalLayout_8->addWidget(rhoEdit);

        rhoScaleComboBox = new QComboBox(rhoGroupBox);
        rhoScaleComboBox->addItem(QString());
        rhoScaleComboBox->addItem(QString());
        rhoScaleComboBox->setObjectName(QString::fromUtf8("rhoScaleComboBox"));

        horizontalLayout_8->addWidget(rhoScaleComboBox);


        verticalLayout_44->addLayout(horizontalLayout_8);


        horizontalLayout_19->addWidget(rhoGroupBox);


        verticalLayout_46->addLayout(horizontalLayout_19);

        medOptionsGroupBox = new QGroupBox(inpmediaGroupBox);
        medOptionsGroupBox->setObjectName(QString::fromUtf8("medOptionsGroupBox"));
        verticalLayout_45 = new QVBoxLayout(medOptionsGroupBox);
        verticalLayout_45->setSpacing(6);
        verticalLayout_45->setContentsMargins(11, 11, 11, 11);
        verticalLayout_45->setObjectName(QString::fromUtf8("verticalLayout_45"));
        verticalLayout_35 = new QVBoxLayout();
        verticalLayout_35->setSpacing(6);
        verticalLayout_35->setObjectName(QString::fromUtf8("verticalLayout_35"));
        DCcheckBox = new QCheckBox(medOptionsGroupBox);
        DCcheckBox->setObjectName(QString::fromUtf8("DCcheckBox"));

        verticalLayout_35->addWidget(DCcheckBox);

        ICRUradCheckBox = new QCheckBox(medOptionsGroupBox);
        ICRUradCheckBox->setObjectName(QString::fromUtf8("ICRUradCheckBox"));

        verticalLayout_35->addWidget(ICRUradCheckBox);

        horizontalLayout_10 = new QHBoxLayout();
        horizontalLayout_10->setSpacing(6);
        horizontalLayout_10->setObjectName(QString::fromUtf8("horizontalLayout_10"));
        isGasCheckBox = new QCheckBox(medOptionsGroupBox);
        isGasCheckBox->setObjectName(QString::fromUtf8("isGasCheckBox"));

        horizontalLayout_10->addWidget(isGasCheckBox);

        horizontalSpacer_5 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_10->addItem(horizontalSpacer_5);

        gaspLabel = new QLabel(medOptionsGroupBox);
        gaspLabel->setObjectName(QString::fromUtf8("gaspLabel"));
        gaspLabel->setWordWrap(false);

        horizontalLayout_10->addWidget(gaspLabel);

        gaspEdit = new QLineEdit(medOptionsGroupBox);
        gaspEdit->setObjectName(QString::fromUtf8("gaspEdit"));

        horizontalLayout_10->addWidget(gaspEdit);

        gaspUnits = new QLabel(medOptionsGroupBox);
        gaspUnits->setObjectName(QString::fromUtf8("gaspUnits"));
        gaspUnits->setWordWrap(false);

        horizontalLayout_10->addWidget(gaspUnits);


        verticalLayout_35->addLayout(horizontalLayout_10);


        verticalLayout_45->addLayout(verticalLayout_35);


        verticalLayout_46->addWidget(medOptionsGroupBox);


        horizontalLayout_24->addLayout(verticalLayout_46);


        verticalLayout_49->addLayout(horizontalLayout_24);

        DFgroupBox = new QGroupBox(inpmediaGroupBox);
        DFgroupBox->setObjectName(QString::fromUtf8("DFgroupBox"));
        verticalLayout_10 = new QVBoxLayout(DFgroupBox);
        verticalLayout_10->setSpacing(6);
        verticalLayout_10->setContentsMargins(11, 11, 11, 11);
        verticalLayout_10->setObjectName(QString::fromUtf8("verticalLayout_10"));
        horizontalLayout_9 = new QHBoxLayout();
        horizontalLayout_9->setSpacing(6);
        horizontalLayout_9->setObjectName(QString::fromUtf8("horizontalLayout_9"));
        DFSearchLabel = new QLabel(DFgroupBox);
        DFSearchLabel->setObjectName(QString::fromUtf8("DFSearchLabel"));

        horizontalLayout_9->addWidget(DFSearchLabel);

        DFSearchComboBox = new QComboBox(DFgroupBox);
        DFSearchComboBox->addItem(QString());
        DFSearchComboBox->addItem(QString());
        DFSearchComboBox->setObjectName(QString::fromUtf8("DFSearchComboBox"));

        horizontalLayout_9->addWidget(DFSearchComboBox);

        hboxLayout38 = new QHBoxLayout();
        hboxLayout38->setSpacing(6);
        hboxLayout38->setObjectName(QString::fromUtf8("hboxLayout38"));
        DFEdit = new QLineEdit(DFgroupBox);
        DFEdit->setObjectName(QString::fromUtf8("DFEdit"));

        hboxLayout38->addWidget(DFEdit);

        hboxLayout39 = new QHBoxLayout();
        hboxLayout39->setSpacing(6);
        hboxLayout39->setObjectName(QString::fromUtf8("hboxLayout39"));
        DFBrowse = new QPushButton(DFgroupBox);
        DFBrowse->setObjectName(QString::fromUtf8("DFBrowse"));

        hboxLayout39->addWidget(DFBrowse);


        hboxLayout38->addLayout(hboxLayout39);


        horizontalLayout_9->addLayout(hboxLayout38);


        verticalLayout_10->addLayout(horizontalLayout_9);


        verticalLayout_49->addWidget(DFgroupBox);


        verticalLayout_50->addLayout(verticalLayout_49);


        verticalLayout_51->addWidget(inpmediaGroupBox);


        verticalLayout_52->addLayout(verticalLayout_51);

        TabWidgetRZ->addTab(MDTab, QString());
        SItab = new QWidget();
        SItab->setObjectName(QString::fromUtf8("SItab"));
        gridLayout_4 = new QGridLayout(SItab);
        gridLayout_4->setSpacing(6);
        gridLayout_4->setContentsMargins(11, 11, 11, 11);
        gridLayout_4->setObjectName(QString::fromUtf8("gridLayout_4"));
        horizontalLayout_11 = new QHBoxLayout();
        horizontalLayout_11->setSpacing(6);
        horizontalLayout_11->setObjectName(QString::fromUtf8("horizontalLayout_11"));
        gridLayout_3 = new QGridLayout();
        gridLayout_3->setSpacing(6);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        verticalLayout_13 = new QVBoxLayout();
        verticalLayout_13->setSpacing(6);
        verticalLayout_13->setObjectName(QString::fromUtf8("verticalLayout_13"));
        in_particleButtonGroup = new QGroupBox(SItab);
        in_particleButtonGroup->setObjectName(QString::fromUtf8("in_particleButtonGroup"));
        in_particleButtonGroup->setMaximumSize(QSize(237, 16777215));
        gridLayout_2 = new QGridLayout(in_particleButtonGroup);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        eRadioButton = new QRadioButton(in_particleButtonGroup);
        eRadioButton->setObjectName(QString::fromUtf8("eRadioButton"));

        gridLayout_2->addWidget(eRadioButton, 0, 0, 1, 1);

        phRadioButton = new QRadioButton(in_particleButtonGroup);
        phRadioButton->setObjectName(QString::fromUtf8("phRadioButton"));
        phRadioButton->setChecked(true);

        gridLayout_2->addWidget(phRadioButton, 1, 0, 1, 1);

        pRadioButton = new QRadioButton(in_particleButtonGroup);
        pRadioButton->setObjectName(QString::fromUtf8("pRadioButton"));

        gridLayout_2->addWidget(pRadioButton, 2, 0, 1, 1);

        chargedRadioButton = new QRadioButton(in_particleButtonGroup);
        chargedRadioButton->setObjectName(QString::fromUtf8("chargedRadioButton"));
        chargedRadioButton->setEnabled(false);

        gridLayout_2->addWidget(chargedRadioButton, 3, 0, 1, 1);

        allRadioButton = new QRadioButton(in_particleButtonGroup);
        allRadioButton->setObjectName(QString::fromUtf8("allRadioButton"));
        allRadioButton->setEnabled(false);

        gridLayout_2->addWidget(allRadioButton, 4, 0, 1, 1);


        verticalLayout_13->addWidget(in_particleButtonGroup);

        energyButtonGroup = new QGroupBox(SItab);
        energyButtonGroup->setObjectName(QString::fromUtf8("energyButtonGroup"));
        gridLayout1 = new QGridLayout(energyButtonGroup);
        gridLayout1->setSpacing(6);
        gridLayout1->setContentsMargins(11, 11, 11, 11);
        gridLayout1->setObjectName(QString::fromUtf8("gridLayout1"));
        monoenergeticRadioButton = new QRadioButton(energyButtonGroup);
        monoenergeticRadioButton->setObjectName(QString::fromUtf8("monoenergeticRadioButton"));
        monoenergeticRadioButton->setChecked(true);

        gridLayout1->addWidget(monoenergeticRadioButton, 0, 0, 1, 1);

        spectrumRadioButton = new QRadioButton(energyButtonGroup);
        spectrumRadioButton->setObjectName(QString::fromUtf8("spectrumRadioButton"));

        gridLayout1->addWidget(spectrumRadioButton, 1, 0, 1, 1);


        verticalLayout_13->addWidget(energyButtonGroup);


        gridLayout_3->addLayout(verticalLayout_13, 0, 0, 1, 1);

        verticalLayout_16 = new QVBoxLayout();
        verticalLayout_16->setSpacing(6);
        verticalLayout_16->setObjectName(QString::fromUtf8("verticalLayout_16"));
        IniEgroupBox = new QGroupBox(SItab);
        IniEgroupBox->setObjectName(QString::fromUtf8("IniEgroupBox"));
        verticalLayout_15 = new QVBoxLayout(IniEgroupBox);
        verticalLayout_15->setSpacing(6);
        verticalLayout_15->setContentsMargins(11, 11, 11, 11);
        verticalLayout_15->setObjectName(QString::fromUtf8("verticalLayout_15"));
        ini_energyEdit = new QLineEdit(IniEgroupBox);
        ini_energyEdit->setObjectName(QString::fromUtf8("ini_energyEdit"));
        sizePolicy4.setHeightForWidth(ini_energyEdit->sizePolicy().hasHeightForWidth());
        ini_energyEdit->setSizePolicy(sizePolicy4);

        verticalLayout_15->addWidget(ini_energyEdit);


        verticalLayout_16->addWidget(IniEgroupBox);

        specfnameGroupBox = new QGroupBox(SItab);
        specfnameGroupBox->setObjectName(QString::fromUtf8("specfnameGroupBox"));
        specfnameGroupBox->setEnabled(false);
        verticalLayout_17 = new QVBoxLayout(specfnameGroupBox);
        verticalLayout_17->setSpacing(6);
        verticalLayout_17->setContentsMargins(11, 11, 11, 11);
        verticalLayout_17->setObjectName(QString::fromUtf8("verticalLayout_17"));
        hboxLayout40 = new QHBoxLayout();
        hboxLayout40->setSpacing(6);
        hboxLayout40->setObjectName(QString::fromUtf8("hboxLayout40"));
        specfnameComboBox = new QComboBox(specfnameGroupBox);
        specfnameComboBox->addItem(QString());
        specfnameComboBox->setObjectName(QString::fromUtf8("specfnameComboBox"));
        specfnameComboBox->setEnabled(false);
        sizePolicy6.setHeightForWidth(specfnameComboBox->sizePolicy().hasHeightForWidth());
        specfnameComboBox->setSizePolicy(sizePolicy6);
        specfnameComboBox->setEditable(true);

        hboxLayout40->addWidget(specfnameComboBox);

        specfnameButton = new QPushButton(specfnameGroupBox);
        specfnameButton->setObjectName(QString::fromUtf8("specfnameButton"));
        sizePolicy7.setHeightForWidth(specfnameButton->sizePolicy().hasHeightForWidth());
        specfnameButton->setSizePolicy(sizePolicy7);

        hboxLayout40->addWidget(specfnameButton);


        verticalLayout_17->addLayout(hboxLayout40);


        verticalLayout_16->addWidget(specfnameGroupBox);

        ioutspGroupBox = new QGroupBox(SItab);
        ioutspGroupBox->setObjectName(QString::fromUtf8("ioutspGroupBox"));
        ioutspGroupBox->setEnabled(false);
        formLayout = new QFormLayout(ioutspGroupBox);
        formLayout->setSpacing(6);
        formLayout->setContentsMargins(11, 11, 11, 11);
        formLayout->setObjectName(QString::fromUtf8("formLayout"));
        ioutspCheckBox = new QCheckBox(ioutspGroupBox);
        ioutspCheckBox->setObjectName(QString::fromUtf8("ioutspCheckBox"));
        ioutspCheckBox->setChecked(true);

        formLayout->setWidget(0, QFormLayout::LabelRole, ioutspCheckBox);


        verticalLayout_16->addWidget(ioutspGroupBox);


        gridLayout_3->addLayout(verticalLayout_16, 1, 0, 1, 1);


        horizontalLayout_11->addLayout(gridLayout_3);

        verticalLayout_14 = new QVBoxLayout();
        verticalLayout_14->setSpacing(6);
        verticalLayout_14->setObjectName(QString::fromUtf8("verticalLayout_14"));
        verticalLayout_12 = new QVBoxLayout();
        verticalLayout_12->setSpacing(6);
        verticalLayout_12->setObjectName(QString::fromUtf8("verticalLayout_12"));
        hboxLayout41 = new QHBoxLayout();
        hboxLayout41->setSpacing(6);
        hboxLayout41->setObjectName(QString::fromUtf8("hboxLayout41"));
        sourceLabel = new QLabel(SItab);
        sourceLabel->setObjectName(QString::fromUtf8("sourceLabel"));
        QSizePolicy sizePolicy14(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy14.setHorizontalStretch(232);
        sizePolicy14.setVerticalStretch(0);
        sizePolicy14.setHeightForWidth(sourceLabel->sizePolicy().hasHeightForWidth());
        sourceLabel->setSizePolicy(sizePolicy14);
        sourceLabel->setWordWrap(false);

        hboxLayout41->addWidget(sourceLabel);

        sourceComboBox = new QComboBox(SItab);
        sourceComboBox->addItem(QString());
        sourceComboBox->addItem(QString());
        sourceComboBox->addItem(QString());
        sourceComboBox->addItem(QString());
        sourceComboBox->addItem(QString());
        sourceComboBox->addItem(QString());
        sourceComboBox->addItem(QString());
        sourceComboBox->addItem(QString());
        sourceComboBox->addItem(QString());
        sourceComboBox->addItem(QString());
        sourceComboBox->addItem(QString());
        sourceComboBox->addItem(QString());
        sourceComboBox->addItem(QString());
        sourceComboBox->addItem(QString());
        sourceComboBox->addItem(QString());
        sourceComboBox->addItem(QString());
        sourceComboBox->setObjectName(QString::fromUtf8("sourceComboBox"));

        hboxLayout41->addWidget(sourceComboBox);


        verticalLayout_12->addLayout(hboxLayout41);

        sourceoptionsGroupBox = new QGroupBox(SItab);
        sourceoptionsGroupBox->setObjectName(QString::fromUtf8("sourceoptionsGroupBox"));
        QFont font1;
        sourceoptionsGroupBox->setFont(font1);
        verticalLayout_11 = new QVBoxLayout(sourceoptionsGroupBox);
        verticalLayout_11->setSpacing(6);
        verticalLayout_11->setContentsMargins(11, 11, 11, 11);
        verticalLayout_11->setObjectName(QString::fromUtf8("verticalLayout_11"));
        vboxLayout28 = new QVBoxLayout();
        vboxLayout28->setSpacing(6);
        vboxLayout28->setObjectName(QString::fromUtf8("vboxLayout28"));
        hboxLayout42 = new QHBoxLayout();
        hboxLayout42->setSpacing(6);
        hboxLayout42->setObjectName(QString::fromUtf8("hboxLayout42"));
        temp1Label = new QLabel(sourceoptionsGroupBox);
        temp1Label->setObjectName(QString::fromUtf8("temp1Label"));
        temp1Label->setWordWrap(false);

        hboxLayout42->addWidget(temp1Label);

        Spacer1_4 = new QSpacerItem(20, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout42->addItem(Spacer1_4);

        temp1Edit = new QLineEdit(sourceoptionsGroupBox);
        temp1Edit->setObjectName(QString::fromUtf8("temp1Edit"));

        hboxLayout42->addWidget(temp1Edit);


        vboxLayout28->addLayout(hboxLayout42);

        hboxLayout43 = new QHBoxLayout();
        hboxLayout43->setSpacing(6);
        hboxLayout43->setObjectName(QString::fromUtf8("hboxLayout43"));
        temp2Label = new QLabel(sourceoptionsGroupBox);
        temp2Label->setObjectName(QString::fromUtf8("temp2Label"));
        temp2Label->setWordWrap(false);

        hboxLayout43->addWidget(temp2Label);

        Spacer2_4 = new QSpacerItem(20, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout43->addItem(Spacer2_4);

        temp2Edit = new QLineEdit(sourceoptionsGroupBox);
        temp2Edit->setObjectName(QString::fromUtf8("temp2Edit"));

        hboxLayout43->addWidget(temp2Edit);


        vboxLayout28->addLayout(hboxLayout43);

        hboxLayout44 = new QHBoxLayout();
        hboxLayout44->setSpacing(6);
        hboxLayout44->setObjectName(QString::fromUtf8("hboxLayout44"));
        temp3Label = new QLabel(sourceoptionsGroupBox);
        temp3Label->setObjectName(QString::fromUtf8("temp3Label"));
        temp3Label->setWordWrap(false);

        hboxLayout44->addWidget(temp3Label);

        Spacer3_3 = new QSpacerItem(20, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout44->addItem(Spacer3_3);

        temp3Edit = new QLineEdit(sourceoptionsGroupBox);
        temp3Edit->setObjectName(QString::fromUtf8("temp3Edit"));

        hboxLayout44->addWidget(temp3Edit);


        vboxLayout28->addLayout(hboxLayout44);

        hboxLayout45 = new QHBoxLayout();
        hboxLayout45->setSpacing(6);
        hboxLayout45->setObjectName(QString::fromUtf8("hboxLayout45"));
        temp4Label = new QLabel(sourceoptionsGroupBox);
        temp4Label->setObjectName(QString::fromUtf8("temp4Label"));
        temp4Label->setWordWrap(false);

        hboxLayout45->addWidget(temp4Label);

        Spacer4_4 = new QSpacerItem(20, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout45->addItem(Spacer4_4);

        temp4Edit = new QLineEdit(sourceoptionsGroupBox);
        temp4Edit->setObjectName(QString::fromUtf8("temp4Edit"));

        hboxLayout45->addWidget(temp4Edit);


        vboxLayout28->addLayout(hboxLayout45);

        imodeComboBox = new QComboBox(sourceoptionsGroupBox);
        imodeComboBox->setObjectName(QString::fromUtf8("imodeComboBox"));

        vboxLayout28->addWidget(imodeComboBox);

        hboxLayout46 = new QHBoxLayout();
        hboxLayout46->setSpacing(6);
        hboxLayout46->setObjectName(QString::fromUtf8("hboxLayout46"));
        temp5Label = new QLabel(sourceoptionsGroupBox);
        temp5Label->setObjectName(QString::fromUtf8("temp5Label"));
        temp5Label->setWordWrap(false);

        hboxLayout46->addWidget(temp5Label);

        Spacer36 = new QSpacerItem(20, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout46->addItem(Spacer36);

        temp5Edit = new QLineEdit(sourceoptionsGroupBox);
        temp5Edit->setObjectName(QString::fromUtf8("temp5Edit"));

        hboxLayout46->addWidget(temp5Edit);


        vboxLayout28->addLayout(hboxLayout46);

        hboxLayout47 = new QHBoxLayout();
        hboxLayout47->setSpacing(6);
        hboxLayout47->setObjectName(QString::fromUtf8("hboxLayout47"));
        temp6Label = new QLabel(sourceoptionsGroupBox);
        temp6Label->setObjectName(QString::fromUtf8("temp6Label"));
        temp6Label->setWordWrap(false);

        hboxLayout47->addWidget(temp6Label);

        Spacer37_2 = new QSpacerItem(20, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout47->addItem(Spacer37_2);

        temp6Edit = new QLineEdit(sourceoptionsGroupBox);
        temp6Edit->setObjectName(QString::fromUtf8("temp6Edit"));

        hboxLayout47->addWidget(temp6Edit);


        vboxLayout28->addLayout(hboxLayout47);

        hboxLayout48 = new QHBoxLayout();
        hboxLayout48->setSpacing(6);
        hboxLayout48->setObjectName(QString::fromUtf8("hboxLayout48"));
        temp7Label = new QLabel(sourceoptionsGroupBox);
        temp7Label->setObjectName(QString::fromUtf8("temp7Label"));
        temp7Label->setWordWrap(false);

        hboxLayout48->addWidget(temp7Label);

        Spacer38_2 = new QSpacerItem(20, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout48->addItem(Spacer38_2);

        temp7Edit = new QLineEdit(sourceoptionsGroupBox);
        temp7Edit->setObjectName(QString::fromUtf8("temp7Edit"));

        hboxLayout48->addWidget(temp7Edit);


        vboxLayout28->addLayout(hboxLayout48);


        verticalLayout_11->addLayout(vboxLayout28);

        verticalSpacer_5 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_11->addItem(verticalSpacer_5);


        verticalLayout_12->addWidget(sourceoptionsGroupBox);


        verticalLayout_14->addLayout(verticalLayout_12);

        phasespaceGroupBox = new QGroupBox(SItab);
        phasespaceGroupBox->setObjectName(QString::fromUtf8("phasespaceGroupBox"));
        phasespaceGroupBox->setEnabled(false);
        verticalLayout_18 = new QVBoxLayout(phasespaceGroupBox);
        verticalLayout_18->setSpacing(6);
        verticalLayout_18->setContentsMargins(11, 11, 11, 11);
        verticalLayout_18->setObjectName(QString::fromUtf8("verticalLayout_18"));
        hboxLayout49 = new QHBoxLayout();
        hboxLayout49->setSpacing(6);
        hboxLayout49->setObjectName(QString::fromUtf8("hboxLayout49"));
        phasespaceComboBox = new QComboBox(phasespaceGroupBox);
        phasespaceComboBox->addItem(QString());
        phasespaceComboBox->setObjectName(QString::fromUtf8("phasespaceComboBox"));
        sizePolicy6.setHeightForWidth(phasespaceComboBox->sizePolicy().hasHeightForWidth());
        phasespaceComboBox->setSizePolicy(sizePolicy6);
        phasespaceComboBox->setEditable(true);

        hboxLayout49->addWidget(phasespaceComboBox);

        phasespacePushButton = new QPushButton(phasespaceGroupBox);
        phasespacePushButton->setObjectName(QString::fromUtf8("phasespacePushButton"));
        sizePolicy7.setHeightForWidth(phasespacePushButton->sizePolicy().hasHeightForWidth());
        phasespacePushButton->setSizePolicy(sizePolicy7);

        hboxLayout49->addWidget(phasespacePushButton);


        verticalLayout_18->addLayout(hboxLayout49);


        verticalLayout_14->addWidget(phasespaceGroupBox);


        horizontalLayout_11->addLayout(verticalLayout_14);

        src20GroupBox = new QGroupBox(SItab);
        src20GroupBox->setObjectName(QString::fromUtf8("src20GroupBox"));
        src20GroupBox->setEnabled(true);
        src20GroupBox->setMaximumSize(QSize(288, 16777215));
        gridLayout_5 = new QGridLayout(src20GroupBox);
        gridLayout_5->setSpacing(6);
        gridLayout_5->setContentsMargins(11, 11, 11, 11);
        gridLayout_5->setObjectName(QString::fromUtf8("gridLayout_5"));
        srcinpmodeButtonGroup = new QGroupBox(src20GroupBox);
        srcinpmodeButtonGroup->setObjectName(QString::fromUtf8("srcinpmodeButtonGroup"));
        srcinpmodeButtonGroup->setEnabled(true);
        hboxLayout50 = new QHBoxLayout(srcinpmodeButtonGroup);
        hboxLayout50->setSpacing(6);
        hboxLayout50->setContentsMargins(11, 11, 11, 11);
        hboxLayout50->setObjectName(QString::fromUtf8("hboxLayout50"));
        hboxLayout51 = new QHBoxLayout();
        hboxLayout51->setSpacing(6);
        hboxLayout51->setObjectName(QString::fromUtf8("hboxLayout51"));
        localRadioButton = new QRadioButton(srcinpmodeButtonGroup);
        localRadioButton->setObjectName(QString::fromUtf8("localRadioButton"));
        localRadioButton->setChecked(true);

        hboxLayout51->addWidget(localRadioButton);

        externalRadioButton = new QRadioButton(srcinpmodeButtonGroup);
        externalRadioButton->setObjectName(QString::fromUtf8("externalRadioButton"));

        hboxLayout51->addWidget(externalRadioButton);


        hboxLayout50->addLayout(hboxLayout51);


        gridLayout_5->addWidget(srcinpmodeButtonGroup, 0, 0, 1, 1);

        spacer56_3 = new QSpacerItem(20, 41, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_5->addItem(spacer56_3, 1, 0, 1, 1);

        raddistTable = new QTableWidget(src20GroupBox);
        if (raddistTable->columnCount() < 2)
            raddistTable->setColumnCount(2);
        QTableWidgetItem *__qtablewidgetitem616 = new QTableWidgetItem();
        raddistTable->setHorizontalHeaderItem(0, __qtablewidgetitem616);
        QTableWidgetItem *__qtablewidgetitem617 = new QTableWidgetItem();
        raddistTable->setHorizontalHeaderItem(1, __qtablewidgetitem617);
        if (raddistTable->rowCount() < 50)
            raddistTable->setRowCount(50);
        QTableWidgetItem *__qtablewidgetitem618 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(0, __qtablewidgetitem618);
        QTableWidgetItem *__qtablewidgetitem619 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(1, __qtablewidgetitem619);
        QTableWidgetItem *__qtablewidgetitem620 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(2, __qtablewidgetitem620);
        QTableWidgetItem *__qtablewidgetitem621 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(3, __qtablewidgetitem621);
        QTableWidgetItem *__qtablewidgetitem622 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(4, __qtablewidgetitem622);
        QTableWidgetItem *__qtablewidgetitem623 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(5, __qtablewidgetitem623);
        QTableWidgetItem *__qtablewidgetitem624 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(6, __qtablewidgetitem624);
        QTableWidgetItem *__qtablewidgetitem625 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(7, __qtablewidgetitem625);
        QTableWidgetItem *__qtablewidgetitem626 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(8, __qtablewidgetitem626);
        QTableWidgetItem *__qtablewidgetitem627 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(9, __qtablewidgetitem627);
        QTableWidgetItem *__qtablewidgetitem628 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(10, __qtablewidgetitem628);
        QTableWidgetItem *__qtablewidgetitem629 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(11, __qtablewidgetitem629);
        QTableWidgetItem *__qtablewidgetitem630 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(12, __qtablewidgetitem630);
        QTableWidgetItem *__qtablewidgetitem631 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(13, __qtablewidgetitem631);
        QTableWidgetItem *__qtablewidgetitem632 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(14, __qtablewidgetitem632);
        QTableWidgetItem *__qtablewidgetitem633 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(15, __qtablewidgetitem633);
        QTableWidgetItem *__qtablewidgetitem634 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(16, __qtablewidgetitem634);
        QTableWidgetItem *__qtablewidgetitem635 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(17, __qtablewidgetitem635);
        QTableWidgetItem *__qtablewidgetitem636 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(18, __qtablewidgetitem636);
        QTableWidgetItem *__qtablewidgetitem637 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(19, __qtablewidgetitem637);
        QTableWidgetItem *__qtablewidgetitem638 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(20, __qtablewidgetitem638);
        QTableWidgetItem *__qtablewidgetitem639 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(21, __qtablewidgetitem639);
        QTableWidgetItem *__qtablewidgetitem640 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(22, __qtablewidgetitem640);
        QTableWidgetItem *__qtablewidgetitem641 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(23, __qtablewidgetitem641);
        QTableWidgetItem *__qtablewidgetitem642 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(24, __qtablewidgetitem642);
        QTableWidgetItem *__qtablewidgetitem643 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(25, __qtablewidgetitem643);
        QTableWidgetItem *__qtablewidgetitem644 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(26, __qtablewidgetitem644);
        QTableWidgetItem *__qtablewidgetitem645 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(27, __qtablewidgetitem645);
        QTableWidgetItem *__qtablewidgetitem646 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(28, __qtablewidgetitem646);
        QTableWidgetItem *__qtablewidgetitem647 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(29, __qtablewidgetitem647);
        QTableWidgetItem *__qtablewidgetitem648 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(30, __qtablewidgetitem648);
        QTableWidgetItem *__qtablewidgetitem649 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(31, __qtablewidgetitem649);
        QTableWidgetItem *__qtablewidgetitem650 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(32, __qtablewidgetitem650);
        QTableWidgetItem *__qtablewidgetitem651 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(33, __qtablewidgetitem651);
        QTableWidgetItem *__qtablewidgetitem652 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(34, __qtablewidgetitem652);
        QTableWidgetItem *__qtablewidgetitem653 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(35, __qtablewidgetitem653);
        QTableWidgetItem *__qtablewidgetitem654 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(36, __qtablewidgetitem654);
        QTableWidgetItem *__qtablewidgetitem655 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(37, __qtablewidgetitem655);
        QTableWidgetItem *__qtablewidgetitem656 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(38, __qtablewidgetitem656);
        QTableWidgetItem *__qtablewidgetitem657 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(39, __qtablewidgetitem657);
        QTableWidgetItem *__qtablewidgetitem658 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(40, __qtablewidgetitem658);
        QTableWidgetItem *__qtablewidgetitem659 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(41, __qtablewidgetitem659);
        QTableWidgetItem *__qtablewidgetitem660 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(42, __qtablewidgetitem660);
        QTableWidgetItem *__qtablewidgetitem661 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(43, __qtablewidgetitem661);
        QTableWidgetItem *__qtablewidgetitem662 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(44, __qtablewidgetitem662);
        QTableWidgetItem *__qtablewidgetitem663 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(45, __qtablewidgetitem663);
        QTableWidgetItem *__qtablewidgetitem664 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(46, __qtablewidgetitem664);
        QTableWidgetItem *__qtablewidgetitem665 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(47, __qtablewidgetitem665);
        QTableWidgetItem *__qtablewidgetitem666 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(48, __qtablewidgetitem666);
        QTableWidgetItem *__qtablewidgetitem667 = new QTableWidgetItem();
        raddistTable->setVerticalHeaderItem(49, __qtablewidgetitem667);
        raddistTable->setObjectName(QString::fromUtf8("raddistTable"));
        raddistTable->setMinimumSize(QSize(250, 132));
        raddistTable->setMaximumSize(QSize(250, 132));
        raddistTable->setRowCount(50);
        raddistTable->setColumnCount(2);

        gridLayout_5->addWidget(raddistTable, 2, 0, 1, 1);

        spacer57_2 = new QSpacerItem(20, 41, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_5->addItem(spacer57_2, 3, 0, 1, 1);

        vboxLayout29 = new QVBoxLayout();
        vboxLayout29->setSpacing(6);
        vboxLayout29->setObjectName(QString::fromUtf8("vboxLayout29"));
        raddistfnameGroupBox = new QGroupBox(src20GroupBox);
        raddistfnameGroupBox->setObjectName(QString::fromUtf8("raddistfnameGroupBox"));
        raddistfnameGroupBox->setEnabled(false);
        hboxLayout52 = new QHBoxLayout(raddistfnameGroupBox);
        hboxLayout52->setSpacing(6);
        hboxLayout52->setContentsMargins(11, 11, 11, 11);
        hboxLayout52->setObjectName(QString::fromUtf8("hboxLayout52"));
        hboxLayout53 = new QHBoxLayout();
        hboxLayout53->setSpacing(6);
        hboxLayout53->setObjectName(QString::fromUtf8("hboxLayout53"));
        raddistfnameComboBox = new QComboBox(raddistfnameGroupBox);
        raddistfnameComboBox->addItem(QString());
        raddistfnameComboBox->setObjectName(QString::fromUtf8("raddistfnameComboBox"));
        sizePolicy6.setHeightForWidth(raddistfnameComboBox->sizePolicy().hasHeightForWidth());
        raddistfnameComboBox->setSizePolicy(sizePolicy6);
        raddistfnameComboBox->setEditable(true);

        hboxLayout53->addWidget(raddistfnameComboBox);

        raddistfnameButton = new QPushButton(raddistfnameGroupBox);
        raddistfnameButton->setObjectName(QString::fromUtf8("raddistfnameButton"));
        sizePolicy7.setHeightForWidth(raddistfnameButton->sizePolicy().hasHeightForWidth());
        raddistfnameButton->setSizePolicy(sizePolicy7);

        hboxLayout53->addWidget(raddistfnameButton);


        hboxLayout52->addLayout(hboxLayout53);


        vboxLayout29->addWidget(raddistfnameGroupBox);

        ioutrdistGroupBox = new QGroupBox(src20GroupBox);
        ioutrdistGroupBox->setObjectName(QString::fromUtf8("ioutrdistGroupBox"));
        ioutrdistGroupBox->setEnabled(false);
        hboxLayout54 = new QHBoxLayout(ioutrdistGroupBox);
        hboxLayout54->setSpacing(6);
        hboxLayout54->setContentsMargins(11, 11, 11, 11);
        hboxLayout54->setObjectName(QString::fromUtf8("hboxLayout54"));
        ioutrdistCheckBox = new QCheckBox(ioutrdistGroupBox);
        ioutrdistCheckBox->setObjectName(QString::fromUtf8("ioutrdistCheckBox"));
        ioutrdistCheckBox->setChecked(true);

        hboxLayout54->addWidget(ioutrdistCheckBox);


        vboxLayout29->addWidget(ioutrdistGroupBox);


        gridLayout_5->addLayout(vboxLayout29, 4, 0, 1, 1);


        horizontalLayout_11->addWidget(src20GroupBox);


        gridLayout_4->addLayout(horizontalLayout_11, 0, 0, 1, 1);

        TabWidgetRZ->addTab(SItab, QString());
        MCTPtab = new QWidget();
        MCTPtab->setObjectName(QString::fromUtf8("MCTPtab"));
        gridLayout_18 = new QGridLayout(MCTPtab);
        gridLayout_18->setSpacing(6);
        gridLayout_18->setContentsMargins(11, 11, 11, 11);
        gridLayout_18->setObjectName(QString::fromUtf8("gridLayout_18"));
        gridLayout_17 = new QGridLayout();
        gridLayout_17->setSpacing(6);
        gridLayout_17->setObjectName(QString::fromUtf8("gridLayout_17"));
        hboxLayout55 = new QHBoxLayout();
        hboxLayout55->setSpacing(6);
        hboxLayout55->setObjectName(QString::fromUtf8("hboxLayout55"));
        vboxLayout30 = new QVBoxLayout();
        vboxLayout30->setSpacing(6);
        vboxLayout30->setObjectName(QString::fromUtf8("vboxLayout30"));
        PairAngSamplingGroupBox = new QGroupBox(MCTPtab);
        PairAngSamplingGroupBox->setObjectName(QString::fromUtf8("PairAngSamplingGroupBox"));
        sizePolicy9.setHeightForWidth(PairAngSamplingGroupBox->sizePolicy().hasHeightForWidth());
        PairAngSamplingGroupBox->setSizePolicy(sizePolicy9);
        PairAngSamplingGroupBox->setFont(font1);
        hboxLayout56 = new QHBoxLayout(PairAngSamplingGroupBox);
        hboxLayout56->setSpacing(6);
        hboxLayout56->setContentsMargins(11, 11, 11, 11);
        hboxLayout56->setObjectName(QString::fromUtf8("hboxLayout56"));
        PairAngSamplingComboBox = new QComboBox(PairAngSamplingGroupBox);
        PairAngSamplingComboBox->addItem(QString());
        PairAngSamplingComboBox->addItem(QString());
        PairAngSamplingComboBox->addItem(QString());
        PairAngSamplingComboBox->setObjectName(QString::fromUtf8("PairAngSamplingComboBox"));

        hboxLayout56->addWidget(PairAngSamplingComboBox);


        vboxLayout30->addWidget(PairAngSamplingGroupBox);

        BremsAngSamplingGroupBox = new QGroupBox(MCTPtab);
        BremsAngSamplingGroupBox->setObjectName(QString::fromUtf8("BremsAngSamplingGroupBox"));
        sizePolicy9.setHeightForWidth(BremsAngSamplingGroupBox->sizePolicy().hasHeightForWidth());
        BremsAngSamplingGroupBox->setSizePolicy(sizePolicy9);
        BremsAngSamplingGroupBox->setFont(font1);
        hboxLayout57 = new QHBoxLayout(BremsAngSamplingGroupBox);
        hboxLayout57->setSpacing(6);
        hboxLayout57->setContentsMargins(11, 11, 11, 11);
        hboxLayout57->setObjectName(QString::fromUtf8("hboxLayout57"));
        BremsAngSamplingComboBox = new QComboBox(BremsAngSamplingGroupBox);
        BremsAngSamplingComboBox->addItem(QString());
        BremsAngSamplingComboBox->addItem(QString());
        BremsAngSamplingComboBox->setObjectName(QString::fromUtf8("BremsAngSamplingComboBox"));

        hboxLayout57->addWidget(BremsAngSamplingComboBox);


        vboxLayout30->addWidget(BremsAngSamplingGroupBox);

        BremsXSectionGroupBox = new QGroupBox(MCTPtab);
        BremsXSectionGroupBox->setObjectName(QString::fromUtf8("BremsXSectionGroupBox"));
        sizePolicy9.setHeightForWidth(BremsXSectionGroupBox->sizePolicy().hasHeightForWidth());
        BremsXSectionGroupBox->setSizePolicy(sizePolicy9);
        BremsXSectionGroupBox->setFont(font1);
        hboxLayout58 = new QHBoxLayout(BremsXSectionGroupBox);
        hboxLayout58->setSpacing(6);
        hboxLayout58->setContentsMargins(11, 11, 11, 11);
        hboxLayout58->setObjectName(QString::fromUtf8("hboxLayout58"));
        BremsXSectionComboBox = new QComboBox(BremsXSectionGroupBox);
        BremsXSectionComboBox->addItem(QString());
        BremsXSectionComboBox->addItem(QString());
        BremsXSectionComboBox->setObjectName(QString::fromUtf8("BremsXSectionComboBox"));

        hboxLayout58->addWidget(BremsXSectionComboBox);


        vboxLayout30->addWidget(BremsXSectionGroupBox);


        hboxLayout55->addLayout(vboxLayout30);

        vboxLayout31 = new QVBoxLayout();
        vboxLayout31->setSpacing(6);
        vboxLayout31->setObjectName(QString::fromUtf8("vboxLayout31"));
        estep_algorithmGroupBox = new QGroupBox(MCTPtab);
        estep_algorithmGroupBox->setObjectName(QString::fromUtf8("estep_algorithmGroupBox"));
        sizePolicy9.setHeightForWidth(estep_algorithmGroupBox->sizePolicy().hasHeightForWidth());
        estep_algorithmGroupBox->setSizePolicy(sizePolicy9);
        estep_algorithmGroupBox->setFont(font1);
        hboxLayout59 = new QHBoxLayout(estep_algorithmGroupBox);
        hboxLayout59->setSpacing(6);
        hboxLayout59->setContentsMargins(11, 11, 11, 11);
        hboxLayout59->setObjectName(QString::fromUtf8("hboxLayout59"));
        estep_algorithmComboBox = new QComboBox(estep_algorithmGroupBox);
        estep_algorithmComboBox->addItem(QString());
        estep_algorithmComboBox->addItem(QString());
        estep_algorithmComboBox->setObjectName(QString::fromUtf8("estep_algorithmComboBox"));

        hboxLayout59->addWidget(estep_algorithmComboBox);


        vboxLayout31->addWidget(estep_algorithmGroupBox);

        EIIgroupBox = new QGroupBox(MCTPtab);
        EIIgroupBox->setObjectName(QString::fromUtf8("EIIgroupBox"));
        sizePolicy9.setHeightForWidth(EIIgroupBox->sizePolicy().hasHeightForWidth());
        EIIgroupBox->setSizePolicy(sizePolicy9);
        EIIgroupBox->setFont(font1);
        vboxLayout32 = new QVBoxLayout(EIIgroupBox);
        vboxLayout32->setSpacing(6);
        vboxLayout32->setContentsMargins(11, 11, 11, 11);
        vboxLayout32->setObjectName(QString::fromUtf8("vboxLayout32"));
        EIIcomboBox = new QComboBox(EIIgroupBox);
        EIIcomboBox->addItem(QString());
        EIIcomboBox->addItem(QString());
        EIIcomboBox->addItem(QString());
        EIIcomboBox->addItem(QString());
        EIIcomboBox->addItem(QString());
        EIIcomboBox->setObjectName(QString::fromUtf8("EIIcomboBox"));
        EIIcomboBox->setEditable(false);

        vboxLayout32->addWidget(EIIcomboBox);


        vboxLayout31->addWidget(EIIgroupBox);

        BCAGroupBox = new QGroupBox(MCTPtab);
        BCAGroupBox->setObjectName(QString::fromUtf8("BCAGroupBox"));
        sizePolicy9.setHeightForWidth(BCAGroupBox->sizePolicy().hasHeightForWidth());
        BCAGroupBox->setSizePolicy(sizePolicy9);
        BCAGroupBox->setFont(font1);
        hboxLayout60 = new QHBoxLayout(BCAGroupBox);
        hboxLayout60->setSpacing(6);
        hboxLayout60->setContentsMargins(11, 11, 11, 11);
        hboxLayout60->setObjectName(QString::fromUtf8("hboxLayout60"));
        BCAComboBox = new QComboBox(BCAGroupBox);
        BCAComboBox->addItem(QString());
        BCAComboBox->addItem(QString());
        BCAComboBox->setObjectName(QString::fromUtf8("BCAComboBox"));

        hboxLayout60->addWidget(BCAComboBox);


        vboxLayout31->addWidget(BCAGroupBox);


        hboxLayout55->addLayout(vboxLayout31);


        gridLayout_17->addLayout(hboxLayout55, 0, 0, 1, 1);

        verticalLayout_20 = new QVBoxLayout();
        verticalLayout_20->setSpacing(6);
        verticalLayout_20->setObjectName(QString::fromUtf8("verticalLayout_20"));
        GECUTGroupBox = new QGroupBox(MCTPtab);
        GECUTGroupBox->setObjectName(QString::fromUtf8("GECUTGroupBox"));
        sizePolicy5.setHeightForWidth(GECUTGroupBox->sizePolicy().hasHeightForWidth());
        GECUTGroupBox->setSizePolicy(sizePolicy5);
        GECUTGroupBox->setFont(font1);
        gridLayout_8 = new QGridLayout(GECUTGroupBox);
        gridLayout_8->setSpacing(6);
        gridLayout_8->setContentsMargins(11, 11, 11, 11);
        gridLayout_8->setObjectName(QString::fromUtf8("gridLayout_8"));
        GECUTEdit = new QLineEdit(GECUTGroupBox);
        GECUTEdit->setObjectName(QString::fromUtf8("GECUTEdit"));

        gridLayout_8->addWidget(GECUTEdit, 0, 0, 1, 1);


        verticalLayout_20->addWidget(GECUTGroupBox);

        GPCUTGroupBox = new QGroupBox(MCTPtab);
        GPCUTGroupBox->setObjectName(QString::fromUtf8("GPCUTGroupBox"));
        sizePolicy5.setHeightForWidth(GPCUTGroupBox->sizePolicy().hasHeightForWidth());
        GPCUTGroupBox->setSizePolicy(sizePolicy5);
        GPCUTGroupBox->setMinimumSize(QSize(115, 0));
        GPCUTGroupBox->setFont(font1);
        gridLayout_15 = new QGridLayout(GPCUTGroupBox);
        gridLayout_15->setSpacing(6);
        gridLayout_15->setContentsMargins(11, 11, 11, 11);
        gridLayout_15->setObjectName(QString::fromUtf8("gridLayout_15"));
        GPCUTEdit = new QLineEdit(GPCUTGroupBox);
        GPCUTEdit->setObjectName(QString::fromUtf8("GPCUTEdit"));
        sizePolicy5.setHeightForWidth(GPCUTEdit->sizePolicy().hasHeightForWidth());
        GPCUTEdit->setSizePolicy(sizePolicy5);
        GPCUTEdit->setMinimumSize(QSize(110, 0));

        gridLayout_15->addWidget(GPCUTEdit, 0, 0, 1, 1);


        verticalLayout_20->addWidget(GPCUTGroupBox);

        GSMAXGroupBox = new QGroupBox(MCTPtab);
        GSMAXGroupBox->setObjectName(QString::fromUtf8("GSMAXGroupBox"));
        sizePolicy5.setHeightForWidth(GSMAXGroupBox->sizePolicy().hasHeightForWidth());
        GSMAXGroupBox->setSizePolicy(sizePolicy5);
        GSMAXGroupBox->setFont(font1);
        gridLayout_14 = new QGridLayout(GSMAXGroupBox);
        gridLayout_14->setSpacing(6);
        gridLayout_14->setContentsMargins(11, 11, 11, 11);
        gridLayout_14->setObjectName(QString::fromUtf8("gridLayout_14"));
        GSMAXEdit = new QLineEdit(GSMAXGroupBox);
        GSMAXEdit->setObjectName(QString::fromUtf8("GSMAXEdit"));

        gridLayout_14->addWidget(GSMAXEdit, 0, 0, 1, 1);


        verticalLayout_20->addWidget(GSMAXGroupBox);


        gridLayout_17->addLayout(verticalLayout_20, 0, 1, 1, 1);

        gridLayout_16 = new QGridLayout();
        gridLayout_16->setSpacing(6);
        gridLayout_16->setObjectName(QString::fromUtf8("gridLayout_16"));
        gridLayout_13 = new QGridLayout();
        gridLayout_13->setSpacing(6);
        gridLayout_13->setObjectName(QString::fromUtf8("gridLayout_13"));
        RayleighgroupBox = new QGroupBox(MCTPtab);
        RayleighgroupBox->setObjectName(QString::fromUtf8("RayleighgroupBox"));
        sizePolicy5.setHeightForWidth(RayleighgroupBox->sizePolicy().hasHeightForWidth());
        RayleighgroupBox->setSizePolicy(sizePolicy5);
        RayleighgroupBox->setMinimumSize(QSize(190, 0));
        RayleighgroupBox->setFont(font1);
        gridLayout_9 = new QGridLayout(RayleighgroupBox);
        gridLayout_9->setSpacing(6);
        gridLayout_9->setContentsMargins(11, 11, 11, 11);
        gridLayout_9->setObjectName(QString::fromUtf8("gridLayout_9"));
        RayleighcomboBox = new QComboBox(RayleighgroupBox);
        RayleighcomboBox->addItem(QString());
        RayleighcomboBox->addItem(QString());
        RayleighcomboBox->addItem(QString());
        RayleighcomboBox->addItem(QString());
        RayleighcomboBox->addItem(QString());
        RayleighcomboBox->setObjectName(QString::fromUtf8("RayleighcomboBox"));

        gridLayout_9->addWidget(RayleighcomboBox, 0, 0, 1, 1);


        gridLayout_13->addWidget(RayleighgroupBox, 0, 0, 1, 1);

        PEgroupBox = new QGroupBox(MCTPtab);
        PEgroupBox->setObjectName(QString::fromUtf8("PEgroupBox"));
        sizePolicy5.setHeightForWidth(PEgroupBox->sizePolicy().hasHeightForWidth());
        PEgroupBox->setSizePolicy(sizePolicy5);
        PEgroupBox->setMinimumSize(QSize(190, 0));
        PEgroupBox->setFont(font1);
        gridLayout_11 = new QGridLayout(PEgroupBox);
        gridLayout_11->setSpacing(6);
        gridLayout_11->setContentsMargins(11, 11, 11, 11);
        gridLayout_11->setObjectName(QString::fromUtf8("gridLayout_11"));
        PEcomboBox = new QComboBox(PEgroupBox);
        PEcomboBox->addItem(QString());
        PEcomboBox->addItem(QString());
        PEcomboBox->addItem(QString());
        PEcomboBox->addItem(QString());
        PEcomboBox->setObjectName(QString::fromUtf8("PEcomboBox"));

        gridLayout_11->addWidget(PEcomboBox, 0, 0, 1, 1);


        gridLayout_13->addWidget(PEgroupBox, 0, 1, 1, 1);

        BoundComptongroupBox = new QGroupBox(MCTPtab);
        BoundComptongroupBox->setObjectName(QString::fromUtf8("BoundComptongroupBox"));
        sizePolicy5.setHeightForWidth(BoundComptongroupBox->sizePolicy().hasHeightForWidth());
        BoundComptongroupBox->setSizePolicy(sizePolicy5);
        BoundComptongroupBox->setFont(font1);
        gridLayout_6 = new QGridLayout(BoundComptongroupBox);
        gridLayout_6->setSpacing(6);
        gridLayout_6->setContentsMargins(11, 11, 11, 11);
        gridLayout_6->setObjectName(QString::fromUtf8("gridLayout_6"));
        BoundComptoncomboBox = new QComboBox(BoundComptongroupBox);
        BoundComptoncomboBox->addItem(QString());
        BoundComptoncomboBox->addItem(QString());
        BoundComptoncomboBox->addItem(QString());
        BoundComptoncomboBox->addItem(QString());
        BoundComptoncomboBox->addItem(QString());
        BoundComptoncomboBox->addItem(QString());
        BoundComptoncomboBox->setObjectName(QString::fromUtf8("BoundComptoncomboBox"));

        gridLayout_6->addWidget(BoundComptoncomboBox, 0, 0, 1, 1);


        gridLayout_13->addWidget(BoundComptongroupBox, 1, 0, 1, 1);

        RelaxationgroupBox = new QGroupBox(MCTPtab);
        RelaxationgroupBox->setObjectName(QString::fromUtf8("RelaxationgroupBox"));
        sizePolicy5.setHeightForWidth(RelaxationgroupBox->sizePolicy().hasHeightForWidth());
        RelaxationgroupBox->setSizePolicy(sizePolicy5);
        RelaxationgroupBox->setMinimumSize(QSize(190, 0));
        RelaxationgroupBox->setFont(font1);
        gridLayout_10 = new QGridLayout(RelaxationgroupBox);
        gridLayout_10->setSpacing(6);
        gridLayout_10->setContentsMargins(11, 11, 11, 11);
        gridLayout_10->setObjectName(QString::fromUtf8("gridLayout_10"));
        RelaxationcomboBox = new QComboBox(RelaxationgroupBox);
        RelaxationcomboBox->addItem(QString());
        RelaxationcomboBox->addItem(QString());
        RelaxationcomboBox->addItem(QString());
        RelaxationcomboBox->addItem(QString());
        RelaxationcomboBox->addItem(QString());
        RelaxationcomboBox->addItem(QString());
        RelaxationcomboBox->setObjectName(QString::fromUtf8("RelaxationcomboBox"));

        gridLayout_10->addWidget(RelaxationcomboBox, 0, 0, 1, 1);


        gridLayout_13->addWidget(RelaxationgroupBox, 1, 1, 1, 1);


        gridLayout_16->addLayout(gridLayout_13, 0, 0, 1, 2);

        verticalLayout_19 = new QVBoxLayout();
        verticalLayout_19->setSpacing(6);
        verticalLayout_19->setObjectName(QString::fromUtf8("verticalLayout_19"));
        ECUTCheckBox = new QCheckBox(MCTPtab);
        ECUTCheckBox->setObjectName(QString::fromUtf8("ECUTCheckBox"));
        ECUTCheckBox->setFont(font1);

        verticalLayout_19->addWidget(ECUTCheckBox);

        PCUTCheckBox = new QCheckBox(MCTPtab);
        PCUTCheckBox->setObjectName(QString::fromUtf8("PCUTCheckBox"));
        PCUTCheckBox->setFont(font1);

        verticalLayout_19->addWidget(PCUTCheckBox);

        SMAXCheckBox = new QCheckBox(MCTPtab);
        SMAXCheckBox->setObjectName(QString::fromUtf8("SMAXCheckBox"));
        SMAXCheckBox->setFont(font1);

        verticalLayout_19->addWidget(SMAXCheckBox);


        gridLayout_16->addLayout(verticalLayout_19, 1, 0, 1, 1);

        vboxLayout33 = new QVBoxLayout();
        vboxLayout33->setSpacing(6);
        vboxLayout33->setObjectName(QString::fromUtf8("vboxLayout33"));
        photonXSectiongroupBox = new QGroupBox(MCTPtab);
        photonXSectiongroupBox->setObjectName(QString::fromUtf8("photonXSectiongroupBox"));
        sizePolicy5.setHeightForWidth(photonXSectiongroupBox->sizePolicy().hasHeightForWidth());
        photonXSectiongroupBox->setSizePolicy(sizePolicy5);
        photonXSectiongroupBox->setFont(font1);
        vboxLayout34 = new QVBoxLayout(photonXSectiongroupBox);
        vboxLayout34->setSpacing(6);
        vboxLayout34->setContentsMargins(11, 11, 11, 11);
        vboxLayout34->setObjectName(QString::fromUtf8("vboxLayout34"));
        photonXSectioncomboBox = new QComboBox(photonXSectiongroupBox);
        photonXSectioncomboBox->addItem(QString());
        photonXSectioncomboBox->addItem(QString());
        photonXSectioncomboBox->addItem(QString());
        photonXSectioncomboBox->setObjectName(QString::fromUtf8("photonXSectioncomboBox"));
        photonXSectioncomboBox->setFont(font1);
        photonXSectioncomboBox->setEditable(false);

        vboxLayout34->addWidget(photonXSectioncomboBox);


        vboxLayout33->addWidget(photonXSectiongroupBox);

        photonXSectionOutCheckBox = new QCheckBox(MCTPtab);
        photonXSectionOutCheckBox->setObjectName(QString::fromUtf8("photonXSectionOutCheckBox"));
        sizePolicy5.setHeightForWidth(photonXSectionOutCheckBox->sizePolicy().hasHeightForWidth());
        photonXSectionOutCheckBox->setSizePolicy(sizePolicy5);
        photonXSectionOutCheckBox->setFont(font1);
        photonXSectionOutCheckBox->setChecked(false);

        vboxLayout33->addWidget(photonXSectionOutCheckBox);


        gridLayout_16->addLayout(vboxLayout33, 1, 1, 1, 1);


        gridLayout_17->addLayout(gridLayout_16, 0, 2, 1, 1);

        horizontalLayout_27 = new QHBoxLayout();
        horizontalLayout_27->setSpacing(6);
        horizontalLayout_27->setObjectName(QString::fromUtf8("horizontalLayout_27"));
        customFFgroupBox = new QGroupBox(MCTPtab);
        customFFgroupBox->setObjectName(QString::fromUtf8("customFFgroupBox"));
        customFFgroupBox->setEnabled(true);
        sizePolicy2.setHeightForWidth(customFFgroupBox->sizePolicy().hasHeightForWidth());
        customFFgroupBox->setSizePolicy(sizePolicy2);
        customFFgroupBox->setMinimumSize(QSize(0, 0));
        gridLayout_12 = new QGridLayout(customFFgroupBox);
        gridLayout_12->setSpacing(6);
        gridLayout_12->setContentsMargins(11, 11, 11, 11);
        gridLayout_12->setObjectName(QString::fromUtf8("gridLayout_12"));
        customFFTable = new QTableWidget(customFFgroupBox);
        if (customFFTable->columnCount() < 2)
            customFFTable->setColumnCount(2);
        QTableWidgetItem *__qtablewidgetitem668 = new QTableWidgetItem();
        customFFTable->setHorizontalHeaderItem(0, __qtablewidgetitem668);
        QTableWidgetItem *__qtablewidgetitem669 = new QTableWidgetItem();
        customFFTable->setHorizontalHeaderItem(1, __qtablewidgetitem669);
        if (customFFTable->rowCount() < 50)
            customFFTable->setRowCount(50);
        QTableWidgetItem *__qtablewidgetitem670 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(0, __qtablewidgetitem670);
        QTableWidgetItem *__qtablewidgetitem671 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(1, __qtablewidgetitem671);
        QTableWidgetItem *__qtablewidgetitem672 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(2, __qtablewidgetitem672);
        QTableWidgetItem *__qtablewidgetitem673 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(3, __qtablewidgetitem673);
        QTableWidgetItem *__qtablewidgetitem674 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(4, __qtablewidgetitem674);
        QTableWidgetItem *__qtablewidgetitem675 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(5, __qtablewidgetitem675);
        QTableWidgetItem *__qtablewidgetitem676 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(6, __qtablewidgetitem676);
        QTableWidgetItem *__qtablewidgetitem677 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(7, __qtablewidgetitem677);
        QTableWidgetItem *__qtablewidgetitem678 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(8, __qtablewidgetitem678);
        QTableWidgetItem *__qtablewidgetitem679 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(9, __qtablewidgetitem679);
        QTableWidgetItem *__qtablewidgetitem680 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(10, __qtablewidgetitem680);
        QTableWidgetItem *__qtablewidgetitem681 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(11, __qtablewidgetitem681);
        QTableWidgetItem *__qtablewidgetitem682 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(12, __qtablewidgetitem682);
        QTableWidgetItem *__qtablewidgetitem683 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(13, __qtablewidgetitem683);
        QTableWidgetItem *__qtablewidgetitem684 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(14, __qtablewidgetitem684);
        QTableWidgetItem *__qtablewidgetitem685 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(15, __qtablewidgetitem685);
        QTableWidgetItem *__qtablewidgetitem686 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(16, __qtablewidgetitem686);
        QTableWidgetItem *__qtablewidgetitem687 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(17, __qtablewidgetitem687);
        QTableWidgetItem *__qtablewidgetitem688 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(18, __qtablewidgetitem688);
        QTableWidgetItem *__qtablewidgetitem689 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(19, __qtablewidgetitem689);
        QTableWidgetItem *__qtablewidgetitem690 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(20, __qtablewidgetitem690);
        QTableWidgetItem *__qtablewidgetitem691 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(21, __qtablewidgetitem691);
        QTableWidgetItem *__qtablewidgetitem692 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(22, __qtablewidgetitem692);
        QTableWidgetItem *__qtablewidgetitem693 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(23, __qtablewidgetitem693);
        QTableWidgetItem *__qtablewidgetitem694 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(24, __qtablewidgetitem694);
        QTableWidgetItem *__qtablewidgetitem695 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(25, __qtablewidgetitem695);
        QTableWidgetItem *__qtablewidgetitem696 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(26, __qtablewidgetitem696);
        QTableWidgetItem *__qtablewidgetitem697 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(27, __qtablewidgetitem697);
        QTableWidgetItem *__qtablewidgetitem698 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(28, __qtablewidgetitem698);
        QTableWidgetItem *__qtablewidgetitem699 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(29, __qtablewidgetitem699);
        QTableWidgetItem *__qtablewidgetitem700 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(30, __qtablewidgetitem700);
        QTableWidgetItem *__qtablewidgetitem701 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(31, __qtablewidgetitem701);
        QTableWidgetItem *__qtablewidgetitem702 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(32, __qtablewidgetitem702);
        QTableWidgetItem *__qtablewidgetitem703 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(33, __qtablewidgetitem703);
        QTableWidgetItem *__qtablewidgetitem704 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(34, __qtablewidgetitem704);
        QTableWidgetItem *__qtablewidgetitem705 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(35, __qtablewidgetitem705);
        QTableWidgetItem *__qtablewidgetitem706 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(36, __qtablewidgetitem706);
        QTableWidgetItem *__qtablewidgetitem707 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(37, __qtablewidgetitem707);
        QTableWidgetItem *__qtablewidgetitem708 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(38, __qtablewidgetitem708);
        QTableWidgetItem *__qtablewidgetitem709 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(39, __qtablewidgetitem709);
        QTableWidgetItem *__qtablewidgetitem710 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(40, __qtablewidgetitem710);
        QTableWidgetItem *__qtablewidgetitem711 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(41, __qtablewidgetitem711);
        QTableWidgetItem *__qtablewidgetitem712 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(42, __qtablewidgetitem712);
        QTableWidgetItem *__qtablewidgetitem713 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(43, __qtablewidgetitem713);
        QTableWidgetItem *__qtablewidgetitem714 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(44, __qtablewidgetitem714);
        QTableWidgetItem *__qtablewidgetitem715 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(45, __qtablewidgetitem715);
        QTableWidgetItem *__qtablewidgetitem716 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(46, __qtablewidgetitem716);
        QTableWidgetItem *__qtablewidgetitem717 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(47, __qtablewidgetitem717);
        QTableWidgetItem *__qtablewidgetitem718 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(48, __qtablewidgetitem718);
        QTableWidgetItem *__qtablewidgetitem719 = new QTableWidgetItem();
        customFFTable->setVerticalHeaderItem(49, __qtablewidgetitem719);
        customFFTable->setObjectName(QString::fromUtf8("customFFTable"));
        customFFTable->setMinimumSize(QSize(570, 174));
        customFFTable->setAutoScroll(true);
        customFFTable->setRowCount(50);
        customFFTable->setColumnCount(2);

        gridLayout_12->addWidget(customFFTable, 0, 0, 1, 1);


        horizontalLayout_27->addWidget(customFFgroupBox);

        verticalLayout_27 = new QVBoxLayout();
        verticalLayout_27->setSpacing(6);
        verticalLayout_27->setObjectName(QString::fromUtf8("verticalLayout_27"));
        vboxLayout35 = new QVBoxLayout();
        vboxLayout35->setSpacing(6);
        vboxLayout35->setObjectName(QString::fromUtf8("vboxLayout35"));
        vboxLayout36 = new QVBoxLayout();
        vboxLayout36->setSpacing(6);
        vboxLayout36->setObjectName(QString::fromUtf8("vboxLayout36"));
        hboxLayout61 = new QHBoxLayout();
        hboxLayout61->setSpacing(6);
        hboxLayout61->setObjectName(QString::fromUtf8("hboxLayout61"));
        ESTEPELabel = new QLabel(MCTPtab);
        ESTEPELabel->setObjectName(QString::fromUtf8("ESTEPELabel"));
        sizePolicy2.setHeightForWidth(ESTEPELabel->sizePolicy().hasHeightForWidth());
        ESTEPELabel->setSizePolicy(sizePolicy2);
        ESTEPELabel->setFont(font1);
        ESTEPELabel->setWordWrap(false);

        hboxLayout61->addWidget(ESTEPELabel);

        ESTEPEEdit = new QLineEdit(MCTPtab);
        ESTEPEEdit->setObjectName(QString::fromUtf8("ESTEPEEdit"));
        sizePolicy5.setHeightForWidth(ESTEPEEdit->sizePolicy().hasHeightForWidth());
        ESTEPEEdit->setSizePolicy(sizePolicy5);
        ESTEPEEdit->setMaximumSize(QSize(90, 32767));
        ESTEPEEdit->setFont(font1);

        hboxLayout61->addWidget(ESTEPEEdit);


        vboxLayout36->addLayout(hboxLayout61);

        hboxLayout62 = new QHBoxLayout();
        hboxLayout62->setSpacing(6);
        hboxLayout62->setObjectName(QString::fromUtf8("hboxLayout62"));
        XImaxLabel = new QLabel(MCTPtab);
        XImaxLabel->setObjectName(QString::fromUtf8("XImaxLabel"));
        XImaxLabel->setFont(font1);
        XImaxLabel->setWordWrap(false);

        hboxLayout62->addWidget(XImaxLabel);

        XImaxEdit = new QLineEdit(MCTPtab);
        XImaxEdit->setObjectName(QString::fromUtf8("XImaxEdit"));
        sizePolicy5.setHeightForWidth(XImaxEdit->sizePolicy().hasHeightForWidth());
        XImaxEdit->setSizePolicy(sizePolicy5);
        XImaxEdit->setMaximumSize(QSize(90, 32767));
        XImaxEdit->setFont(font1);

        hboxLayout62->addWidget(XImaxEdit);


        vboxLayout36->addLayout(hboxLayout62);

        hboxLayout63 = new QHBoxLayout();
        hboxLayout63->setSpacing(6);
        hboxLayout63->setObjectName(QString::fromUtf8("hboxLayout63"));
        SkinDepthLabel = new QLabel(MCTPtab);
        SkinDepthLabel->setObjectName(QString::fromUtf8("SkinDepthLabel"));
        SkinDepthLabel->setFont(font1);
        SkinDepthLabel->setWordWrap(false);

        hboxLayout63->addWidget(SkinDepthLabel);

        SkinDepthEdit = new QLineEdit(MCTPtab);
        SkinDepthEdit->setObjectName(QString::fromUtf8("SkinDepthEdit"));
        sizePolicy5.setHeightForWidth(SkinDepthEdit->sizePolicy().hasHeightForWidth());
        SkinDepthEdit->setSizePolicy(sizePolicy5);
        SkinDepthEdit->setMaximumSize(QSize(90, 32767));
        SkinDepthEdit->setFont(font1);

        hboxLayout63->addWidget(SkinDepthEdit);


        vboxLayout36->addLayout(hboxLayout63);


        vboxLayout35->addLayout(vboxLayout36);

        SpinCheckBox = new QCheckBox(MCTPtab);
        SpinCheckBox->setObjectName(QString::fromUtf8("SpinCheckBox"));
        sizePolicy5.setHeightForWidth(SpinCheckBox->sizePolicy().hasHeightForWidth());
        SpinCheckBox->setSizePolicy(sizePolicy5);
        SpinCheckBox->setFont(font1);
        SpinCheckBox->setChecked(true);

        vboxLayout35->addWidget(SpinCheckBox);


        verticalLayout_27->addLayout(vboxLayout35);

        verticalSpacer_4 = new QSpacerItem(20, 58, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_27->addItem(verticalSpacer_4);


        horizontalLayout_27->addLayout(verticalLayout_27);


        gridLayout_17->addLayout(horizontalLayout_27, 1, 0, 1, 3);


        gridLayout_18->addLayout(gridLayout_17, 0, 0, 1, 1);

        TabWidgetRZ->addTab(MCTPtab, QString());
        VRtab = new QWidget();
        VRtab->setObjectName(QString::fromUtf8("VRtab"));
        layoutWidget3 = new QWidget(VRtab);
        layoutWidget3->setObjectName(QString::fromUtf8("layoutWidget3"));
        layoutWidget3->setGeometry(QRect(10, 0, 921, 541));
        horizontalLayout_22 = new QHBoxLayout(layoutWidget3);
        horizontalLayout_22->setSpacing(6);
        horizontalLayout_22->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_22->setObjectName(QString::fromUtf8("horizontalLayout_22"));
        horizontalLayout_22->setContentsMargins(0, 0, 0, 0);
        vboxLayout37 = new QVBoxLayout();
        vboxLayout37->setSpacing(6);
        vboxLayout37->setObjectName(QString::fromUtf8("vboxLayout37"));
        PhotonForcingGroupBox = new QGroupBox(layoutWidget3);
        PhotonForcingGroupBox->setObjectName(QString::fromUtf8("PhotonForcingGroupBox"));
        vboxLayout38 = new QVBoxLayout(PhotonForcingGroupBox);
        vboxLayout38->setSpacing(6);
        vboxLayout38->setContentsMargins(11, 11, 11, 11);
        vboxLayout38->setObjectName(QString::fromUtf8("vboxLayout38"));
        vboxLayout39 = new QVBoxLayout();
        vboxLayout39->setSpacing(6);
        vboxLayout39->setObjectName(QString::fromUtf8("vboxLayout39"));
        PhotonForcingCheckBox = new QCheckBox(PhotonForcingGroupBox);
        PhotonForcingCheckBox->setObjectName(QString::fromUtf8("PhotonForcingCheckBox"));
        sizePolicy13.setHeightForWidth(PhotonForcingCheckBox->sizePolicy().hasHeightForWidth());
        PhotonForcingCheckBox->setSizePolicy(sizePolicy13);
        QFont font2;
        font2.setFamily(QString::fromUtf8("Helvetica"));
        PhotonForcingCheckBox->setFont(font2);

        vboxLayout39->addWidget(PhotonForcingCheckBox);

        hboxLayout64 = new QHBoxLayout();
        hboxLayout64->setSpacing(6);
        hboxLayout64->setObjectName(QString::fromUtf8("hboxLayout64"));
        vboxLayout40 = new QVBoxLayout();
        vboxLayout40->setSpacing(6);
        vboxLayout40->setObjectName(QString::fromUtf8("vboxLayout40"));
        StartForcingLabel = new QLabel(PhotonForcingGroupBox);
        StartForcingLabel->setObjectName(QString::fromUtf8("StartForcingLabel"));
        StartForcingLabel->setEnabled(false);
        StartForcingLabel->setWordWrap(false);

        vboxLayout40->addWidget(StartForcingLabel);

        StopForcingLabel = new QLabel(PhotonForcingGroupBox);
        StopForcingLabel->setObjectName(QString::fromUtf8("StopForcingLabel"));
        StopForcingLabel->setEnabled(false);
        StopForcingLabel->setWordWrap(false);

        vboxLayout40->addWidget(StopForcingLabel);


        hboxLayout64->addLayout(vboxLayout40);

        vboxLayout41 = new QVBoxLayout();
        vboxLayout41->setSpacing(6);
        vboxLayout41->setObjectName(QString::fromUtf8("vboxLayout41"));
        StartForcingSpinBox = new QSpinBox(PhotonForcingGroupBox);
        StartForcingSpinBox->setObjectName(QString::fromUtf8("StartForcingSpinBox"));
        StartForcingSpinBox->setEnabled(false);

        vboxLayout41->addWidget(StartForcingSpinBox);

        StopForcingSpinBox = new QSpinBox(PhotonForcingGroupBox);
        StopForcingSpinBox->setObjectName(QString::fromUtf8("StopForcingSpinBox"));
        StopForcingSpinBox->setEnabled(false);

        vboxLayout41->addWidget(StopForcingSpinBox);


        hboxLayout64->addLayout(vboxLayout41);


        vboxLayout39->addLayout(hboxLayout64);


        vboxLayout38->addLayout(vboxLayout39);


        vboxLayout37->addWidget(PhotonForcingGroupBox);

        ExpTrafoCGroupBox = new QGroupBox(layoutWidget3);
        ExpTrafoCGroupBox->setObjectName(QString::fromUtf8("ExpTrafoCGroupBox"));
        hboxLayout65 = new QHBoxLayout(ExpTrafoCGroupBox);
        hboxLayout65->setSpacing(6);
        hboxLayout65->setContentsMargins(11, 11, 11, 11);
        hboxLayout65->setObjectName(QString::fromUtf8("hboxLayout65"));
        hboxLayout66 = new QHBoxLayout();
        hboxLayout66->setSpacing(6);
        hboxLayout66->setObjectName(QString::fromUtf8("hboxLayout66"));
        ExpTrafoCLabel = new QLabel(ExpTrafoCGroupBox);
        ExpTrafoCLabel->setObjectName(QString::fromUtf8("ExpTrafoCLabel"));
        ExpTrafoCLabel->setWordWrap(false);

        hboxLayout66->addWidget(ExpTrafoCLabel);

        ExpTrafoCEdit = new QLineEdit(ExpTrafoCGroupBox);
        ExpTrafoCEdit->setObjectName(QString::fromUtf8("ExpTrafoCEdit"));

        hboxLayout66->addWidget(ExpTrafoCEdit);


        hboxLayout65->addLayout(hboxLayout66);


        vboxLayout37->addWidget(ExpTrafoCGroupBox);

        RRGroupBox = new QGroupBox(layoutWidget3);
        RRGroupBox->setObjectName(QString::fromUtf8("RRGroupBox"));
        vboxLayout42 = new QVBoxLayout(RRGroupBox);
        vboxLayout42->setSpacing(6);
        vboxLayout42->setContentsMargins(11, 11, 11, 11);
        vboxLayout42->setObjectName(QString::fromUtf8("vboxLayout42"));
        vboxLayout43 = new QVBoxLayout();
        vboxLayout43->setSpacing(6);
        vboxLayout43->setObjectName(QString::fromUtf8("vboxLayout43"));
        hboxLayout67 = new QHBoxLayout();
        hboxLayout67->setSpacing(6);
        hboxLayout67->setObjectName(QString::fromUtf8("hboxLayout67"));
        RRDepthLabel = new QLabel(RRGroupBox);
        RRDepthLabel->setObjectName(QString::fromUtf8("RRDepthLabel"));
        RRDepthLabel->setWordWrap(false);

        hboxLayout67->addWidget(RRDepthLabel);

        RRDepthEdit = new QLineEdit(RRGroupBox);
        RRDepthEdit->setObjectName(QString::fromUtf8("RRDepthEdit"));

        hboxLayout67->addWidget(RRDepthEdit);


        vboxLayout43->addLayout(hboxLayout67);

        hboxLayout68 = new QHBoxLayout();
        hboxLayout68->setSpacing(6);
        hboxLayout68->setObjectName(QString::fromUtf8("hboxLayout68"));
        RRFractionLabel = new QLabel(RRGroupBox);
        RRFractionLabel->setObjectName(QString::fromUtf8("RRFractionLabel"));
        RRFractionLabel->setWordWrap(false);

        hboxLayout68->addWidget(RRFractionLabel);

        RRFractionEdit = new QLineEdit(RRGroupBox);
        RRFractionEdit->setObjectName(QString::fromUtf8("RRFractionEdit"));

        hboxLayout68->addWidget(RRFractionEdit);


        vboxLayout43->addLayout(hboxLayout68);


        vboxLayout42->addLayout(vboxLayout43);


        vboxLayout37->addWidget(RRGroupBox);

        photonSplitGroupBox = new QGroupBox(layoutWidget3);
        photonSplitGroupBox->setObjectName(QString::fromUtf8("photonSplitGroupBox"));
        hboxLayout69 = new QHBoxLayout(photonSplitGroupBox);
        hboxLayout69->setSpacing(6);
        hboxLayout69->setContentsMargins(11, 11, 11, 11);
        hboxLayout69->setObjectName(QString::fromUtf8("hboxLayout69"));
        hboxLayout70 = new QHBoxLayout();
        hboxLayout70->setSpacing(6);
        hboxLayout70->setObjectName(QString::fromUtf8("hboxLayout70"));
        photonSplitSpinBox = new QSpinBox(photonSplitGroupBox);
        photonSplitSpinBox->setObjectName(QString::fromUtf8("photonSplitSpinBox"));
        photonSplitSpinBox->setMinimum(1);
        photonSplitSpinBox->setMaximum(10000);
        photonSplitSpinBox->setValue(1);

        hboxLayout70->addWidget(photonSplitSpinBox);

        photonSplitLabel = new QLabel(photonSplitGroupBox);
        photonSplitLabel->setObjectName(QString::fromUtf8("photonSplitLabel"));
        photonSplitLabel->setWordWrap(false);

        hboxLayout70->addWidget(photonSplitLabel);


        hboxLayout69->addLayout(hboxLayout70);


        vboxLayout37->addWidget(photonSplitGroupBox);


        horizontalLayout_22->addLayout(vboxLayout37);

        vboxLayout44 = new QVBoxLayout();
        vboxLayout44->setSpacing(6);
        vboxLayout44->setObjectName(QString::fromUtf8("vboxLayout44"));
        CSEnhancementGroupBox = new QGroupBox(layoutWidget3);
        CSEnhancementGroupBox->setObjectName(QString::fromUtf8("CSEnhancementGroupBox"));
        layoutWidget4 = new QWidget(CSEnhancementGroupBox);
        layoutWidget4->setObjectName(QString::fromUtf8("layoutWidget4"));
        layoutWidget4->setGeometry(QRect(19, 32, 371, 231));
        verticalLayout_2 = new QVBoxLayout(layoutWidget4);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        TextLabel8 = new QLabel(layoutWidget4);
        TextLabel8->setObjectName(QString::fromUtf8("TextLabel8"));
        QSizePolicy sizePolicy15(QSizePolicy::MinimumExpanding, QSizePolicy::Preferred);
        sizePolicy15.setHorizontalStretch(0);
        sizePolicy15.setVerticalStretch(0);
        sizePolicy15.setHeightForWidth(TextLabel8->sizePolicy().hasHeightForWidth());
        TextLabel8->setSizePolicy(sizePolicy15);
        TextLabel8->setWordWrap(false);

        horizontalLayout_2->addWidget(TextLabel8);

        CSEnhancementSpinBox = new QSpinBox(layoutWidget4);
        CSEnhancementSpinBox->setObjectName(QString::fromUtf8("CSEnhancementSpinBox"));
        CSEnhancementSpinBox->setMinimum(1);
        CSEnhancementSpinBox->setMaximum(10000);
        CSEnhancementSpinBox->setValue(200);

        horizontalLayout_2->addWidget(CSEnhancementSpinBox);


        verticalLayout_2->addLayout(horizontalLayout_2);

        vboxLayout45 = new QVBoxLayout();
        vboxLayout45->setSpacing(6);
        vboxLayout45->setObjectName(QString::fromUtf8("vboxLayout45"));
        vboxLayout46 = new QVBoxLayout();
        vboxLayout46->setSpacing(6);
        vboxLayout46->setObjectName(QString::fromUtf8("vboxLayout46"));
        CSEnhancement_RegionsLabel1 = new QLabel(layoutWidget4);
        CSEnhancement_RegionsLabel1->setObjectName(QString::fromUtf8("CSEnhancement_RegionsLabel1"));
        CSEnhancement_RegionsLabel1->setEnabled(false);
        CSEnhancement_RegionsLabel1->setWordWrap(false);

        vboxLayout46->addWidget(CSEnhancement_RegionsLabel1);

        CSEnhancement_RegionsLabel2 = new QLabel(layoutWidget4);
        CSEnhancement_RegionsLabel2->setObjectName(QString::fromUtf8("CSEnhancement_RegionsLabel2"));
        CSEnhancement_RegionsLabel2->setEnabled(false);
        CSEnhancement_RegionsLabel2->setAlignment(Qt::AlignCenter);
        CSEnhancement_RegionsLabel2->setWordWrap(false);

        vboxLayout46->addWidget(CSEnhancement_RegionsLabel2);


        vboxLayout45->addLayout(vboxLayout46);

        hboxLayout71 = new QHBoxLayout();
        hboxLayout71->setSpacing(6);
        hboxLayout71->setObjectName(QString::fromUtf8("hboxLayout71"));
        CSEnhancementTable = new QTableWidget(layoutWidget4);
        if (CSEnhancementTable->columnCount() < 2)
            CSEnhancementTable->setColumnCount(2);
        QTableWidgetItem *__qtablewidgetitem720 = new QTableWidgetItem();
        CSEnhancementTable->setHorizontalHeaderItem(0, __qtablewidgetitem720);
        QTableWidgetItem *__qtablewidgetitem721 = new QTableWidgetItem();
        CSEnhancementTable->setHorizontalHeaderItem(1, __qtablewidgetitem721);
        if (CSEnhancementTable->rowCount() < 50)
            CSEnhancementTable->setRowCount(50);
        QTableWidgetItem *__qtablewidgetitem722 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(0, __qtablewidgetitem722);
        QTableWidgetItem *__qtablewidgetitem723 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(1, __qtablewidgetitem723);
        QTableWidgetItem *__qtablewidgetitem724 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(2, __qtablewidgetitem724);
        QTableWidgetItem *__qtablewidgetitem725 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(3, __qtablewidgetitem725);
        QTableWidgetItem *__qtablewidgetitem726 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(4, __qtablewidgetitem726);
        QTableWidgetItem *__qtablewidgetitem727 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(5, __qtablewidgetitem727);
        QTableWidgetItem *__qtablewidgetitem728 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(6, __qtablewidgetitem728);
        QTableWidgetItem *__qtablewidgetitem729 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(7, __qtablewidgetitem729);
        QTableWidgetItem *__qtablewidgetitem730 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(8, __qtablewidgetitem730);
        QTableWidgetItem *__qtablewidgetitem731 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(9, __qtablewidgetitem731);
        QTableWidgetItem *__qtablewidgetitem732 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(10, __qtablewidgetitem732);
        QTableWidgetItem *__qtablewidgetitem733 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(11, __qtablewidgetitem733);
        QTableWidgetItem *__qtablewidgetitem734 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(12, __qtablewidgetitem734);
        QTableWidgetItem *__qtablewidgetitem735 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(13, __qtablewidgetitem735);
        QTableWidgetItem *__qtablewidgetitem736 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(14, __qtablewidgetitem736);
        QTableWidgetItem *__qtablewidgetitem737 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(15, __qtablewidgetitem737);
        QTableWidgetItem *__qtablewidgetitem738 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(16, __qtablewidgetitem738);
        QTableWidgetItem *__qtablewidgetitem739 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(17, __qtablewidgetitem739);
        QTableWidgetItem *__qtablewidgetitem740 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(18, __qtablewidgetitem740);
        QTableWidgetItem *__qtablewidgetitem741 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(19, __qtablewidgetitem741);
        QTableWidgetItem *__qtablewidgetitem742 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(20, __qtablewidgetitem742);
        QTableWidgetItem *__qtablewidgetitem743 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(21, __qtablewidgetitem743);
        QTableWidgetItem *__qtablewidgetitem744 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(22, __qtablewidgetitem744);
        QTableWidgetItem *__qtablewidgetitem745 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(23, __qtablewidgetitem745);
        QTableWidgetItem *__qtablewidgetitem746 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(24, __qtablewidgetitem746);
        QTableWidgetItem *__qtablewidgetitem747 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(25, __qtablewidgetitem747);
        QTableWidgetItem *__qtablewidgetitem748 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(26, __qtablewidgetitem748);
        QTableWidgetItem *__qtablewidgetitem749 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(27, __qtablewidgetitem749);
        QTableWidgetItem *__qtablewidgetitem750 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(28, __qtablewidgetitem750);
        QTableWidgetItem *__qtablewidgetitem751 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(29, __qtablewidgetitem751);
        QTableWidgetItem *__qtablewidgetitem752 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(30, __qtablewidgetitem752);
        QTableWidgetItem *__qtablewidgetitem753 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(31, __qtablewidgetitem753);
        QTableWidgetItem *__qtablewidgetitem754 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(32, __qtablewidgetitem754);
        QTableWidgetItem *__qtablewidgetitem755 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(33, __qtablewidgetitem755);
        QTableWidgetItem *__qtablewidgetitem756 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(34, __qtablewidgetitem756);
        QTableWidgetItem *__qtablewidgetitem757 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(35, __qtablewidgetitem757);
        QTableWidgetItem *__qtablewidgetitem758 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(36, __qtablewidgetitem758);
        QTableWidgetItem *__qtablewidgetitem759 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(37, __qtablewidgetitem759);
        QTableWidgetItem *__qtablewidgetitem760 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(38, __qtablewidgetitem760);
        QTableWidgetItem *__qtablewidgetitem761 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(39, __qtablewidgetitem761);
        QTableWidgetItem *__qtablewidgetitem762 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(40, __qtablewidgetitem762);
        QTableWidgetItem *__qtablewidgetitem763 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(41, __qtablewidgetitem763);
        QTableWidgetItem *__qtablewidgetitem764 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(42, __qtablewidgetitem764);
        QTableWidgetItem *__qtablewidgetitem765 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(43, __qtablewidgetitem765);
        QTableWidgetItem *__qtablewidgetitem766 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(44, __qtablewidgetitem766);
        QTableWidgetItem *__qtablewidgetitem767 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(45, __qtablewidgetitem767);
        QTableWidgetItem *__qtablewidgetitem768 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(46, __qtablewidgetitem768);
        QTableWidgetItem *__qtablewidgetitem769 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(47, __qtablewidgetitem769);
        QTableWidgetItem *__qtablewidgetitem770 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(48, __qtablewidgetitem770);
        QTableWidgetItem *__qtablewidgetitem771 = new QTableWidgetItem();
        CSEnhancementTable->setVerticalHeaderItem(49, __qtablewidgetitem771);
        CSEnhancementTable->setObjectName(QString::fromUtf8("CSEnhancementTable"));
        CSEnhancementTable->setMinimumSize(QSize(190, 0));
        CSEnhancementTable->setMaximumSize(QSize(195, 115));
        CSEnhancementTable->setRowCount(50);
        CSEnhancementTable->setColumnCount(2);

        hboxLayout71->addWidget(CSEnhancementTable);


        vboxLayout45->addLayout(hboxLayout71);


        verticalLayout_2->addLayout(vboxLayout45);


        vboxLayout44->addWidget(CSEnhancementGroupBox);

        vboxLayout47 = new QVBoxLayout();
        vboxLayout47->setSpacing(6);
        vboxLayout47->setObjectName(QString::fromUtf8("vboxLayout47"));
        ESAVEINGroupBox = new QGroupBox(layoutWidget3);
        ESAVEINGroupBox->setObjectName(QString::fromUtf8("ESAVEINGroupBox"));
        vboxLayout48 = new QVBoxLayout(ESAVEINGroupBox);
        vboxLayout48->setSpacing(6);
        vboxLayout48->setContentsMargins(11, 11, 11, 11);
        vboxLayout48->setObjectName(QString::fromUtf8("vboxLayout48"));
        vboxLayout49 = new QVBoxLayout();
        vboxLayout49->setSpacing(6);
        vboxLayout49->setObjectName(QString::fromUtf8("vboxLayout49"));
        eRangeRejCheckBox = new QCheckBox(ESAVEINGroupBox);
        eRangeRejCheckBox->setObjectName(QString::fromUtf8("eRangeRejCheckBox"));

        vboxLayout49->addWidget(eRangeRejCheckBox);

        hboxLayout72 = new QHBoxLayout();
        hboxLayout72->setSpacing(6);
        hboxLayout72->setObjectName(QString::fromUtf8("hboxLayout72"));
        ESAVEINLabel = new QLabel(ESAVEINGroupBox);
        ESAVEINLabel->setObjectName(QString::fromUtf8("ESAVEINLabel"));
        ESAVEINLabel->setEnabled(false);
        ESAVEINLabel->setWordWrap(false);

        hboxLayout72->addWidget(ESAVEINLabel);

        Spacer74 = new QSpacerItem(20, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout72->addItem(Spacer74);

        ESAVEINEdit = new QLineEdit(ESAVEINGroupBox);
        ESAVEINEdit->setObjectName(QString::fromUtf8("ESAVEINEdit"));
        ESAVEINEdit->setEnabled(false);

        hboxLayout72->addWidget(ESAVEINEdit);


        vboxLayout49->addLayout(hboxLayout72);


        vboxLayout48->addLayout(vboxLayout49);


        vboxLayout47->addWidget(ESAVEINGroupBox);

        BremsSplitGroupBox = new QGroupBox(layoutWidget3);
        BremsSplitGroupBox->setObjectName(QString::fromUtf8("BremsSplitGroupBox"));
        vboxLayout50 = new QVBoxLayout(BremsSplitGroupBox);
        vboxLayout50->setSpacing(6);
        vboxLayout50->setContentsMargins(11, 11, 11, 11);
        vboxLayout50->setObjectName(QString::fromUtf8("vboxLayout50"));
        vboxLayout51 = new QVBoxLayout();
        vboxLayout51->setSpacing(6);
        vboxLayout51->setObjectName(QString::fromUtf8("vboxLayout51"));
        hboxLayout73 = new QHBoxLayout();
        hboxLayout73->setSpacing(6);
        hboxLayout73->setObjectName(QString::fromUtf8("hboxLayout73"));
        BremsSplitSpinBox = new QSpinBox(BremsSplitGroupBox);
        BremsSplitSpinBox->setObjectName(QString::fromUtf8("BremsSplitSpinBox"));
        BremsSplitSpinBox->setMinimum(1);
        BremsSplitSpinBox->setMaximum(1000);
        BremsSplitSpinBox->setValue(1);

        hboxLayout73->addWidget(BremsSplitSpinBox);

        BremsSplitTextLabel = new QLabel(BremsSplitGroupBox);
        BremsSplitTextLabel->setObjectName(QString::fromUtf8("BremsSplitTextLabel"));
        BremsSplitTextLabel->setWordWrap(false);

        hboxLayout73->addWidget(BremsSplitTextLabel);


        vboxLayout51->addLayout(hboxLayout73);

        BremsSplitCheckBox = new QCheckBox(BremsSplitGroupBox);
        BremsSplitCheckBox->setObjectName(QString::fromUtf8("BremsSplitCheckBox"));

        vboxLayout51->addWidget(BremsSplitCheckBox);

        ChargedPartRRCheckBox = new QCheckBox(BremsSplitGroupBox);
        ChargedPartRRCheckBox->setObjectName(QString::fromUtf8("ChargedPartRRCheckBox"));

        vboxLayout51->addWidget(ChargedPartRRCheckBox);


        vboxLayout50->addLayout(vboxLayout51);


        vboxLayout47->addWidget(BremsSplitGroupBox);


        vboxLayout44->addLayout(vboxLayout47);


        horizontalLayout_22->addLayout(vboxLayout44);

        TabWidgetRZ->addTab(VRtab, QString());
        MCTPRegTab = new QWidget();
        MCTPRegTab->setObjectName(QString::fromUtf8("MCTPRegTab"));
        layoutWidget5 = new QWidget(MCTPRegTab);
        layoutWidget5->setObjectName(QString::fromUtf8("layoutWidget5"));
        layoutWidget5->setGeometry(QRect(20, 3, 901, 541));
        verticalLayout_43 = new QVBoxLayout(layoutWidget5);
        verticalLayout_43->setSpacing(6);
        verticalLayout_43->setContentsMargins(11, 11, 11, 11);
        verticalLayout_43->setObjectName(QString::fromUtf8("verticalLayout_43"));
        verticalLayout_43->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_26 = new QHBoxLayout();
        horizontalLayout_26->setSpacing(6);
        horizontalLayout_26->setObjectName(QString::fromUtf8("horizontalLayout_26"));
        PCUTGroupBox = new QGroupBox(layoutWidget5);
        PCUTGroupBox->setObjectName(QString::fromUtf8("PCUTGroupBox"));
        verticalLayout_40 = new QVBoxLayout(PCUTGroupBox);
        verticalLayout_40->setSpacing(6);
        verticalLayout_40->setContentsMargins(11, 11, 11, 11);
        verticalLayout_40->setObjectName(QString::fromUtf8("verticalLayout_40"));
        PCUTTable = new QTableWidget(PCUTGroupBox);
        if (PCUTTable->columnCount() < 3)
            PCUTTable->setColumnCount(3);
        QTableWidgetItem *__qtablewidgetitem772 = new QTableWidgetItem();
        PCUTTable->setHorizontalHeaderItem(0, __qtablewidgetitem772);
        QTableWidgetItem *__qtablewidgetitem773 = new QTableWidgetItem();
        PCUTTable->setHorizontalHeaderItem(1, __qtablewidgetitem773);
        QTableWidgetItem *__qtablewidgetitem774 = new QTableWidgetItem();
        PCUTTable->setHorizontalHeaderItem(2, __qtablewidgetitem774);
        if (PCUTTable->rowCount() < 20)
            PCUTTable->setRowCount(20);
        QTableWidgetItem *__qtablewidgetitem775 = new QTableWidgetItem();
        PCUTTable->setVerticalHeaderItem(0, __qtablewidgetitem775);
        QTableWidgetItem *__qtablewidgetitem776 = new QTableWidgetItem();
        PCUTTable->setVerticalHeaderItem(1, __qtablewidgetitem776);
        QTableWidgetItem *__qtablewidgetitem777 = new QTableWidgetItem();
        PCUTTable->setVerticalHeaderItem(2, __qtablewidgetitem777);
        QTableWidgetItem *__qtablewidgetitem778 = new QTableWidgetItem();
        PCUTTable->setVerticalHeaderItem(3, __qtablewidgetitem778);
        QTableWidgetItem *__qtablewidgetitem779 = new QTableWidgetItem();
        PCUTTable->setVerticalHeaderItem(4, __qtablewidgetitem779);
        QTableWidgetItem *__qtablewidgetitem780 = new QTableWidgetItem();
        PCUTTable->setVerticalHeaderItem(5, __qtablewidgetitem780);
        QTableWidgetItem *__qtablewidgetitem781 = new QTableWidgetItem();
        PCUTTable->setVerticalHeaderItem(6, __qtablewidgetitem781);
        QTableWidgetItem *__qtablewidgetitem782 = new QTableWidgetItem();
        PCUTTable->setVerticalHeaderItem(7, __qtablewidgetitem782);
        QTableWidgetItem *__qtablewidgetitem783 = new QTableWidgetItem();
        PCUTTable->setVerticalHeaderItem(8, __qtablewidgetitem783);
        QTableWidgetItem *__qtablewidgetitem784 = new QTableWidgetItem();
        PCUTTable->setVerticalHeaderItem(9, __qtablewidgetitem784);
        QTableWidgetItem *__qtablewidgetitem785 = new QTableWidgetItem();
        PCUTTable->setVerticalHeaderItem(10, __qtablewidgetitem785);
        QTableWidgetItem *__qtablewidgetitem786 = new QTableWidgetItem();
        PCUTTable->setVerticalHeaderItem(11, __qtablewidgetitem786);
        QTableWidgetItem *__qtablewidgetitem787 = new QTableWidgetItem();
        PCUTTable->setVerticalHeaderItem(12, __qtablewidgetitem787);
        QTableWidgetItem *__qtablewidgetitem788 = new QTableWidgetItem();
        PCUTTable->setVerticalHeaderItem(13, __qtablewidgetitem788);
        QTableWidgetItem *__qtablewidgetitem789 = new QTableWidgetItem();
        PCUTTable->setVerticalHeaderItem(14, __qtablewidgetitem789);
        QTableWidgetItem *__qtablewidgetitem790 = new QTableWidgetItem();
        PCUTTable->setVerticalHeaderItem(15, __qtablewidgetitem790);
        QTableWidgetItem *__qtablewidgetitem791 = new QTableWidgetItem();
        PCUTTable->setVerticalHeaderItem(16, __qtablewidgetitem791);
        QTableWidgetItem *__qtablewidgetitem792 = new QTableWidgetItem();
        PCUTTable->setVerticalHeaderItem(17, __qtablewidgetitem792);
        QTableWidgetItem *__qtablewidgetitem793 = new QTableWidgetItem();
        PCUTTable->setVerticalHeaderItem(18, __qtablewidgetitem793);
        QTableWidgetItem *__qtablewidgetitem794 = new QTableWidgetItem();
        PCUTTable->setVerticalHeaderItem(19, __qtablewidgetitem794);
        PCUTTable->setObjectName(QString::fromUtf8("PCUTTable"));
        sizePolicy2.setHeightForWidth(PCUTTable->sizePolicy().hasHeightForWidth());
        PCUTTable->setSizePolicy(sizePolicy2);
        PCUTTable->setRowCount(20);
        PCUTTable->setColumnCount(3);

        verticalLayout_40->addWidget(PCUTTable);


        horizontalLayout_26->addWidget(PCUTGroupBox);

        ECUTGroupBox = new QGroupBox(layoutWidget5);
        ECUTGroupBox->setObjectName(QString::fromUtf8("ECUTGroupBox"));
        verticalLayout_41 = new QVBoxLayout(ECUTGroupBox);
        verticalLayout_41->setSpacing(6);
        verticalLayout_41->setContentsMargins(11, 11, 11, 11);
        verticalLayout_41->setObjectName(QString::fromUtf8("verticalLayout_41"));
        ECUTTable = new QTableWidget(ECUTGroupBox);
        if (ECUTTable->columnCount() < 3)
            ECUTTable->setColumnCount(3);
        QTableWidgetItem *__qtablewidgetitem795 = new QTableWidgetItem();
        ECUTTable->setHorizontalHeaderItem(0, __qtablewidgetitem795);
        QTableWidgetItem *__qtablewidgetitem796 = new QTableWidgetItem();
        ECUTTable->setHorizontalHeaderItem(1, __qtablewidgetitem796);
        QTableWidgetItem *__qtablewidgetitem797 = new QTableWidgetItem();
        ECUTTable->setHorizontalHeaderItem(2, __qtablewidgetitem797);
        if (ECUTTable->rowCount() < 20)
            ECUTTable->setRowCount(20);
        QTableWidgetItem *__qtablewidgetitem798 = new QTableWidgetItem();
        ECUTTable->setVerticalHeaderItem(0, __qtablewidgetitem798);
        QTableWidgetItem *__qtablewidgetitem799 = new QTableWidgetItem();
        ECUTTable->setVerticalHeaderItem(1, __qtablewidgetitem799);
        QTableWidgetItem *__qtablewidgetitem800 = new QTableWidgetItem();
        ECUTTable->setVerticalHeaderItem(2, __qtablewidgetitem800);
        QTableWidgetItem *__qtablewidgetitem801 = new QTableWidgetItem();
        ECUTTable->setVerticalHeaderItem(3, __qtablewidgetitem801);
        QTableWidgetItem *__qtablewidgetitem802 = new QTableWidgetItem();
        ECUTTable->setVerticalHeaderItem(4, __qtablewidgetitem802);
        QTableWidgetItem *__qtablewidgetitem803 = new QTableWidgetItem();
        ECUTTable->setVerticalHeaderItem(5, __qtablewidgetitem803);
        QTableWidgetItem *__qtablewidgetitem804 = new QTableWidgetItem();
        ECUTTable->setVerticalHeaderItem(6, __qtablewidgetitem804);
        QTableWidgetItem *__qtablewidgetitem805 = new QTableWidgetItem();
        ECUTTable->setVerticalHeaderItem(7, __qtablewidgetitem805);
        QTableWidgetItem *__qtablewidgetitem806 = new QTableWidgetItem();
        ECUTTable->setVerticalHeaderItem(8, __qtablewidgetitem806);
        QTableWidgetItem *__qtablewidgetitem807 = new QTableWidgetItem();
        ECUTTable->setVerticalHeaderItem(9, __qtablewidgetitem807);
        QTableWidgetItem *__qtablewidgetitem808 = new QTableWidgetItem();
        ECUTTable->setVerticalHeaderItem(10, __qtablewidgetitem808);
        QTableWidgetItem *__qtablewidgetitem809 = new QTableWidgetItem();
        ECUTTable->setVerticalHeaderItem(11, __qtablewidgetitem809);
        QTableWidgetItem *__qtablewidgetitem810 = new QTableWidgetItem();
        ECUTTable->setVerticalHeaderItem(12, __qtablewidgetitem810);
        QTableWidgetItem *__qtablewidgetitem811 = new QTableWidgetItem();
        ECUTTable->setVerticalHeaderItem(13, __qtablewidgetitem811);
        QTableWidgetItem *__qtablewidgetitem812 = new QTableWidgetItem();
        ECUTTable->setVerticalHeaderItem(14, __qtablewidgetitem812);
        QTableWidgetItem *__qtablewidgetitem813 = new QTableWidgetItem();
        ECUTTable->setVerticalHeaderItem(15, __qtablewidgetitem813);
        QTableWidgetItem *__qtablewidgetitem814 = new QTableWidgetItem();
        ECUTTable->setVerticalHeaderItem(16, __qtablewidgetitem814);
        QTableWidgetItem *__qtablewidgetitem815 = new QTableWidgetItem();
        ECUTTable->setVerticalHeaderItem(17, __qtablewidgetitem815);
        QTableWidgetItem *__qtablewidgetitem816 = new QTableWidgetItem();
        ECUTTable->setVerticalHeaderItem(18, __qtablewidgetitem816);
        QTableWidgetItem *__qtablewidgetitem817 = new QTableWidgetItem();
        ECUTTable->setVerticalHeaderItem(19, __qtablewidgetitem817);
        ECUTTable->setObjectName(QString::fromUtf8("ECUTTable"));
        sizePolicy2.setHeightForWidth(ECUTTable->sizePolicy().hasHeightForWidth());
        ECUTTable->setSizePolicy(sizePolicy2);
        ECUTTable->setRowCount(20);
        ECUTTable->setColumnCount(3);

        verticalLayout_41->addWidget(ECUTTable);


        horizontalLayout_26->addWidget(ECUTGroupBox);

        SMAXGroupBox = new QGroupBox(layoutWidget5);
        SMAXGroupBox->setObjectName(QString::fromUtf8("SMAXGroupBox"));
        verticalLayout_42 = new QVBoxLayout(SMAXGroupBox);
        verticalLayout_42->setSpacing(6);
        verticalLayout_42->setContentsMargins(11, 11, 11, 11);
        verticalLayout_42->setObjectName(QString::fromUtf8("verticalLayout_42"));
        SMAXTable = new QTableWidget(SMAXGroupBox);
        if (SMAXTable->columnCount() < 3)
            SMAXTable->setColumnCount(3);
        QTableWidgetItem *__qtablewidgetitem818 = new QTableWidgetItem();
        SMAXTable->setHorizontalHeaderItem(0, __qtablewidgetitem818);
        QTableWidgetItem *__qtablewidgetitem819 = new QTableWidgetItem();
        SMAXTable->setHorizontalHeaderItem(1, __qtablewidgetitem819);
        QTableWidgetItem *__qtablewidgetitem820 = new QTableWidgetItem();
        SMAXTable->setHorizontalHeaderItem(2, __qtablewidgetitem820);
        if (SMAXTable->rowCount() < 20)
            SMAXTable->setRowCount(20);
        QTableWidgetItem *__qtablewidgetitem821 = new QTableWidgetItem();
        SMAXTable->setVerticalHeaderItem(0, __qtablewidgetitem821);
        QTableWidgetItem *__qtablewidgetitem822 = new QTableWidgetItem();
        SMAXTable->setVerticalHeaderItem(1, __qtablewidgetitem822);
        QTableWidgetItem *__qtablewidgetitem823 = new QTableWidgetItem();
        SMAXTable->setVerticalHeaderItem(2, __qtablewidgetitem823);
        QTableWidgetItem *__qtablewidgetitem824 = new QTableWidgetItem();
        SMAXTable->setVerticalHeaderItem(3, __qtablewidgetitem824);
        QTableWidgetItem *__qtablewidgetitem825 = new QTableWidgetItem();
        SMAXTable->setVerticalHeaderItem(4, __qtablewidgetitem825);
        QTableWidgetItem *__qtablewidgetitem826 = new QTableWidgetItem();
        SMAXTable->setVerticalHeaderItem(5, __qtablewidgetitem826);
        QTableWidgetItem *__qtablewidgetitem827 = new QTableWidgetItem();
        SMAXTable->setVerticalHeaderItem(6, __qtablewidgetitem827);
        QTableWidgetItem *__qtablewidgetitem828 = new QTableWidgetItem();
        SMAXTable->setVerticalHeaderItem(7, __qtablewidgetitem828);
        QTableWidgetItem *__qtablewidgetitem829 = new QTableWidgetItem();
        SMAXTable->setVerticalHeaderItem(8, __qtablewidgetitem829);
        QTableWidgetItem *__qtablewidgetitem830 = new QTableWidgetItem();
        SMAXTable->setVerticalHeaderItem(9, __qtablewidgetitem830);
        QTableWidgetItem *__qtablewidgetitem831 = new QTableWidgetItem();
        SMAXTable->setVerticalHeaderItem(10, __qtablewidgetitem831);
        QTableWidgetItem *__qtablewidgetitem832 = new QTableWidgetItem();
        SMAXTable->setVerticalHeaderItem(11, __qtablewidgetitem832);
        QTableWidgetItem *__qtablewidgetitem833 = new QTableWidgetItem();
        SMAXTable->setVerticalHeaderItem(12, __qtablewidgetitem833);
        QTableWidgetItem *__qtablewidgetitem834 = new QTableWidgetItem();
        SMAXTable->setVerticalHeaderItem(13, __qtablewidgetitem834);
        QTableWidgetItem *__qtablewidgetitem835 = new QTableWidgetItem();
        SMAXTable->setVerticalHeaderItem(14, __qtablewidgetitem835);
        QTableWidgetItem *__qtablewidgetitem836 = new QTableWidgetItem();
        SMAXTable->setVerticalHeaderItem(15, __qtablewidgetitem836);
        QTableWidgetItem *__qtablewidgetitem837 = new QTableWidgetItem();
        SMAXTable->setVerticalHeaderItem(16, __qtablewidgetitem837);
        QTableWidgetItem *__qtablewidgetitem838 = new QTableWidgetItem();
        SMAXTable->setVerticalHeaderItem(17, __qtablewidgetitem838);
        QTableWidgetItem *__qtablewidgetitem839 = new QTableWidgetItem();
        SMAXTable->setVerticalHeaderItem(18, __qtablewidgetitem839);
        QTableWidgetItem *__qtablewidgetitem840 = new QTableWidgetItem();
        SMAXTable->setVerticalHeaderItem(19, __qtablewidgetitem840);
        SMAXTable->setObjectName(QString::fromUtf8("SMAXTable"));
        sizePolicy2.setHeightForWidth(SMAXTable->sizePolicy().hasHeightForWidth());
        SMAXTable->setSizePolicy(sizePolicy2);
        SMAXTable->setRowCount(20);
        SMAXTable->setColumnCount(3);

        verticalLayout_42->addWidget(SMAXTable);


        horizontalLayout_26->addWidget(SMAXGroupBox);


        verticalLayout_43->addLayout(horizontalLayout_26);

        horizontalLayout_25 = new QHBoxLayout();
        horizontalLayout_25->setSpacing(6);
        horizontalLayout_25->setObjectName(QString::fromUtf8("horizontalLayout_25"));
        BoundComptonGroupBox = new QGroupBox(layoutWidget5);
        BoundComptonGroupBox->setObjectName(QString::fromUtf8("BoundComptonGroupBox"));
        verticalLayout_39 = new QVBoxLayout(BoundComptonGroupBox);
        verticalLayout_39->setSpacing(6);
        verticalLayout_39->setContentsMargins(11, 11, 11, 11);
        verticalLayout_39->setObjectName(QString::fromUtf8("verticalLayout_39"));
        BoundComptonTable = new QTableWidget(BoundComptonGroupBox);
        if (BoundComptonTable->columnCount() < 2)
            BoundComptonTable->setColumnCount(2);
        QTableWidgetItem *__qtablewidgetitem841 = new QTableWidgetItem();
        BoundComptonTable->setHorizontalHeaderItem(0, __qtablewidgetitem841);
        QTableWidgetItem *__qtablewidgetitem842 = new QTableWidgetItem();
        BoundComptonTable->setHorizontalHeaderItem(1, __qtablewidgetitem842);
        if (BoundComptonTable->rowCount() < 20)
            BoundComptonTable->setRowCount(20);
        QTableWidgetItem *__qtablewidgetitem843 = new QTableWidgetItem();
        BoundComptonTable->setVerticalHeaderItem(0, __qtablewidgetitem843);
        QTableWidgetItem *__qtablewidgetitem844 = new QTableWidgetItem();
        BoundComptonTable->setVerticalHeaderItem(1, __qtablewidgetitem844);
        QTableWidgetItem *__qtablewidgetitem845 = new QTableWidgetItem();
        BoundComptonTable->setVerticalHeaderItem(2, __qtablewidgetitem845);
        QTableWidgetItem *__qtablewidgetitem846 = new QTableWidgetItem();
        BoundComptonTable->setVerticalHeaderItem(3, __qtablewidgetitem846);
        QTableWidgetItem *__qtablewidgetitem847 = new QTableWidgetItem();
        BoundComptonTable->setVerticalHeaderItem(4, __qtablewidgetitem847);
        QTableWidgetItem *__qtablewidgetitem848 = new QTableWidgetItem();
        BoundComptonTable->setVerticalHeaderItem(5, __qtablewidgetitem848);
        QTableWidgetItem *__qtablewidgetitem849 = new QTableWidgetItem();
        BoundComptonTable->setVerticalHeaderItem(6, __qtablewidgetitem849);
        QTableWidgetItem *__qtablewidgetitem850 = new QTableWidgetItem();
        BoundComptonTable->setVerticalHeaderItem(7, __qtablewidgetitem850);
        QTableWidgetItem *__qtablewidgetitem851 = new QTableWidgetItem();
        BoundComptonTable->setVerticalHeaderItem(8, __qtablewidgetitem851);
        QTableWidgetItem *__qtablewidgetitem852 = new QTableWidgetItem();
        BoundComptonTable->setVerticalHeaderItem(9, __qtablewidgetitem852);
        QTableWidgetItem *__qtablewidgetitem853 = new QTableWidgetItem();
        BoundComptonTable->setVerticalHeaderItem(10, __qtablewidgetitem853);
        QTableWidgetItem *__qtablewidgetitem854 = new QTableWidgetItem();
        BoundComptonTable->setVerticalHeaderItem(11, __qtablewidgetitem854);
        QTableWidgetItem *__qtablewidgetitem855 = new QTableWidgetItem();
        BoundComptonTable->setVerticalHeaderItem(12, __qtablewidgetitem855);
        QTableWidgetItem *__qtablewidgetitem856 = new QTableWidgetItem();
        BoundComptonTable->setVerticalHeaderItem(13, __qtablewidgetitem856);
        QTableWidgetItem *__qtablewidgetitem857 = new QTableWidgetItem();
        BoundComptonTable->setVerticalHeaderItem(14, __qtablewidgetitem857);
        QTableWidgetItem *__qtablewidgetitem858 = new QTableWidgetItem();
        BoundComptonTable->setVerticalHeaderItem(15, __qtablewidgetitem858);
        QTableWidgetItem *__qtablewidgetitem859 = new QTableWidgetItem();
        BoundComptonTable->setVerticalHeaderItem(16, __qtablewidgetitem859);
        QTableWidgetItem *__qtablewidgetitem860 = new QTableWidgetItem();
        BoundComptonTable->setVerticalHeaderItem(17, __qtablewidgetitem860);
        QTableWidgetItem *__qtablewidgetitem861 = new QTableWidgetItem();
        BoundComptonTable->setVerticalHeaderItem(18, __qtablewidgetitem861);
        QTableWidgetItem *__qtablewidgetitem862 = new QTableWidgetItem();
        BoundComptonTable->setVerticalHeaderItem(19, __qtablewidgetitem862);
        BoundComptonTable->setObjectName(QString::fromUtf8("BoundComptonTable"));
        sizePolicy2.setHeightForWidth(BoundComptonTable->sizePolicy().hasHeightForWidth());
        BoundComptonTable->setSizePolicy(sizePolicy2);
        BoundComptonTable->setRowCount(20);
        BoundComptonTable->setColumnCount(2);

        verticalLayout_39->addWidget(BoundComptonTable);


        horizontalLayout_25->addWidget(BoundComptonGroupBox);

        RelaxationsGroupBox = new QGroupBox(layoutWidget5);
        RelaxationsGroupBox->setObjectName(QString::fromUtf8("RelaxationsGroupBox"));
        verticalLayout_38 = new QVBoxLayout(RelaxationsGroupBox);
        verticalLayout_38->setSpacing(6);
        verticalLayout_38->setContentsMargins(11, 11, 11, 11);
        verticalLayout_38->setObjectName(QString::fromUtf8("verticalLayout_38"));
        RelaxationsTable = new QTableWidget(RelaxationsGroupBox);
        if (RelaxationsTable->columnCount() < 2)
            RelaxationsTable->setColumnCount(2);
        QTableWidgetItem *__qtablewidgetitem863 = new QTableWidgetItem();
        RelaxationsTable->setHorizontalHeaderItem(0, __qtablewidgetitem863);
        QTableWidgetItem *__qtablewidgetitem864 = new QTableWidgetItem();
        RelaxationsTable->setHorizontalHeaderItem(1, __qtablewidgetitem864);
        if (RelaxationsTable->rowCount() < 20)
            RelaxationsTable->setRowCount(20);
        QTableWidgetItem *__qtablewidgetitem865 = new QTableWidgetItem();
        RelaxationsTable->setVerticalHeaderItem(0, __qtablewidgetitem865);
        QTableWidgetItem *__qtablewidgetitem866 = new QTableWidgetItem();
        RelaxationsTable->setVerticalHeaderItem(1, __qtablewidgetitem866);
        QTableWidgetItem *__qtablewidgetitem867 = new QTableWidgetItem();
        RelaxationsTable->setVerticalHeaderItem(2, __qtablewidgetitem867);
        QTableWidgetItem *__qtablewidgetitem868 = new QTableWidgetItem();
        RelaxationsTable->setVerticalHeaderItem(3, __qtablewidgetitem868);
        QTableWidgetItem *__qtablewidgetitem869 = new QTableWidgetItem();
        RelaxationsTable->setVerticalHeaderItem(4, __qtablewidgetitem869);
        QTableWidgetItem *__qtablewidgetitem870 = new QTableWidgetItem();
        RelaxationsTable->setVerticalHeaderItem(5, __qtablewidgetitem870);
        QTableWidgetItem *__qtablewidgetitem871 = new QTableWidgetItem();
        RelaxationsTable->setVerticalHeaderItem(6, __qtablewidgetitem871);
        QTableWidgetItem *__qtablewidgetitem872 = new QTableWidgetItem();
        RelaxationsTable->setVerticalHeaderItem(7, __qtablewidgetitem872);
        QTableWidgetItem *__qtablewidgetitem873 = new QTableWidgetItem();
        RelaxationsTable->setVerticalHeaderItem(8, __qtablewidgetitem873);
        QTableWidgetItem *__qtablewidgetitem874 = new QTableWidgetItem();
        RelaxationsTable->setVerticalHeaderItem(9, __qtablewidgetitem874);
        QTableWidgetItem *__qtablewidgetitem875 = new QTableWidgetItem();
        RelaxationsTable->setVerticalHeaderItem(10, __qtablewidgetitem875);
        QTableWidgetItem *__qtablewidgetitem876 = new QTableWidgetItem();
        RelaxationsTable->setVerticalHeaderItem(11, __qtablewidgetitem876);
        QTableWidgetItem *__qtablewidgetitem877 = new QTableWidgetItem();
        RelaxationsTable->setVerticalHeaderItem(12, __qtablewidgetitem877);
        QTableWidgetItem *__qtablewidgetitem878 = new QTableWidgetItem();
        RelaxationsTable->setVerticalHeaderItem(13, __qtablewidgetitem878);
        QTableWidgetItem *__qtablewidgetitem879 = new QTableWidgetItem();
        RelaxationsTable->setVerticalHeaderItem(14, __qtablewidgetitem879);
        QTableWidgetItem *__qtablewidgetitem880 = new QTableWidgetItem();
        RelaxationsTable->setVerticalHeaderItem(15, __qtablewidgetitem880);
        QTableWidgetItem *__qtablewidgetitem881 = new QTableWidgetItem();
        RelaxationsTable->setVerticalHeaderItem(16, __qtablewidgetitem881);
        QTableWidgetItem *__qtablewidgetitem882 = new QTableWidgetItem();
        RelaxationsTable->setVerticalHeaderItem(17, __qtablewidgetitem882);
        QTableWidgetItem *__qtablewidgetitem883 = new QTableWidgetItem();
        RelaxationsTable->setVerticalHeaderItem(18, __qtablewidgetitem883);
        QTableWidgetItem *__qtablewidgetitem884 = new QTableWidgetItem();
        RelaxationsTable->setVerticalHeaderItem(19, __qtablewidgetitem884);
        RelaxationsTable->setObjectName(QString::fromUtf8("RelaxationsTable"));
        sizePolicy2.setHeightForWidth(RelaxationsTable->sizePolicy().hasHeightForWidth());
        RelaxationsTable->setSizePolicy(sizePolicy2);
        RelaxationsTable->setRowCount(20);
        RelaxationsTable->setColumnCount(2);

        verticalLayout_38->addWidget(RelaxationsTable);


        horizontalLayout_25->addWidget(RelaxationsGroupBox);

        PEAngSamGroupBox = new QGroupBox(layoutWidget5);
        PEAngSamGroupBox->setObjectName(QString::fromUtf8("PEAngSamGroupBox"));
        verticalLayout_29 = new QVBoxLayout(PEAngSamGroupBox);
        verticalLayout_29->setSpacing(6);
        verticalLayout_29->setContentsMargins(11, 11, 11, 11);
        verticalLayout_29->setObjectName(QString::fromUtf8("verticalLayout_29"));
        PEAngSamplingTable = new QTableWidget(PEAngSamGroupBox);
        if (PEAngSamplingTable->columnCount() < 2)
            PEAngSamplingTable->setColumnCount(2);
        QTableWidgetItem *__qtablewidgetitem885 = new QTableWidgetItem();
        PEAngSamplingTable->setHorizontalHeaderItem(0, __qtablewidgetitem885);
        QTableWidgetItem *__qtablewidgetitem886 = new QTableWidgetItem();
        PEAngSamplingTable->setHorizontalHeaderItem(1, __qtablewidgetitem886);
        if (PEAngSamplingTable->rowCount() < 20)
            PEAngSamplingTable->setRowCount(20);
        QTableWidgetItem *__qtablewidgetitem887 = new QTableWidgetItem();
        PEAngSamplingTable->setVerticalHeaderItem(0, __qtablewidgetitem887);
        QTableWidgetItem *__qtablewidgetitem888 = new QTableWidgetItem();
        PEAngSamplingTable->setVerticalHeaderItem(1, __qtablewidgetitem888);
        QTableWidgetItem *__qtablewidgetitem889 = new QTableWidgetItem();
        PEAngSamplingTable->setVerticalHeaderItem(2, __qtablewidgetitem889);
        QTableWidgetItem *__qtablewidgetitem890 = new QTableWidgetItem();
        PEAngSamplingTable->setVerticalHeaderItem(3, __qtablewidgetitem890);
        QTableWidgetItem *__qtablewidgetitem891 = new QTableWidgetItem();
        PEAngSamplingTable->setVerticalHeaderItem(4, __qtablewidgetitem891);
        QTableWidgetItem *__qtablewidgetitem892 = new QTableWidgetItem();
        PEAngSamplingTable->setVerticalHeaderItem(5, __qtablewidgetitem892);
        QTableWidgetItem *__qtablewidgetitem893 = new QTableWidgetItem();
        PEAngSamplingTable->setVerticalHeaderItem(6, __qtablewidgetitem893);
        QTableWidgetItem *__qtablewidgetitem894 = new QTableWidgetItem();
        PEAngSamplingTable->setVerticalHeaderItem(7, __qtablewidgetitem894);
        QTableWidgetItem *__qtablewidgetitem895 = new QTableWidgetItem();
        PEAngSamplingTable->setVerticalHeaderItem(8, __qtablewidgetitem895);
        QTableWidgetItem *__qtablewidgetitem896 = new QTableWidgetItem();
        PEAngSamplingTable->setVerticalHeaderItem(9, __qtablewidgetitem896);
        QTableWidgetItem *__qtablewidgetitem897 = new QTableWidgetItem();
        PEAngSamplingTable->setVerticalHeaderItem(10, __qtablewidgetitem897);
        QTableWidgetItem *__qtablewidgetitem898 = new QTableWidgetItem();
        PEAngSamplingTable->setVerticalHeaderItem(11, __qtablewidgetitem898);
        QTableWidgetItem *__qtablewidgetitem899 = new QTableWidgetItem();
        PEAngSamplingTable->setVerticalHeaderItem(12, __qtablewidgetitem899);
        QTableWidgetItem *__qtablewidgetitem900 = new QTableWidgetItem();
        PEAngSamplingTable->setVerticalHeaderItem(13, __qtablewidgetitem900);
        QTableWidgetItem *__qtablewidgetitem901 = new QTableWidgetItem();
        PEAngSamplingTable->setVerticalHeaderItem(14, __qtablewidgetitem901);
        QTableWidgetItem *__qtablewidgetitem902 = new QTableWidgetItem();
        PEAngSamplingTable->setVerticalHeaderItem(15, __qtablewidgetitem902);
        QTableWidgetItem *__qtablewidgetitem903 = new QTableWidgetItem();
        PEAngSamplingTable->setVerticalHeaderItem(16, __qtablewidgetitem903);
        QTableWidgetItem *__qtablewidgetitem904 = new QTableWidgetItem();
        PEAngSamplingTable->setVerticalHeaderItem(17, __qtablewidgetitem904);
        QTableWidgetItem *__qtablewidgetitem905 = new QTableWidgetItem();
        PEAngSamplingTable->setVerticalHeaderItem(18, __qtablewidgetitem905);
        QTableWidgetItem *__qtablewidgetitem906 = new QTableWidgetItem();
        PEAngSamplingTable->setVerticalHeaderItem(19, __qtablewidgetitem906);
        PEAngSamplingTable->setObjectName(QString::fromUtf8("PEAngSamplingTable"));
        sizePolicy2.setHeightForWidth(PEAngSamplingTable->sizePolicy().hasHeightForWidth());
        PEAngSamplingTable->setSizePolicy(sizePolicy2);
        PEAngSamplingTable->setRowCount(20);
        PEAngSamplingTable->setColumnCount(2);

        verticalLayout_29->addWidget(PEAngSamplingTable);


        horizontalLayout_25->addWidget(PEAngSamGroupBox);

        RayleighGroupBox = new QGroupBox(layoutWidget5);
        RayleighGroupBox->setObjectName(QString::fromUtf8("RayleighGroupBox"));
        verticalLayout_32 = new QVBoxLayout(RayleighGroupBox);
        verticalLayout_32->setSpacing(6);
        verticalLayout_32->setContentsMargins(11, 11, 11, 11);
        verticalLayout_32->setObjectName(QString::fromUtf8("verticalLayout_32"));
        RayleighTable = new QTableWidget(RayleighGroupBox);
        if (RayleighTable->columnCount() < 2)
            RayleighTable->setColumnCount(2);
        QTableWidgetItem *__qtablewidgetitem907 = new QTableWidgetItem();
        RayleighTable->setHorizontalHeaderItem(0, __qtablewidgetitem907);
        QTableWidgetItem *__qtablewidgetitem908 = new QTableWidgetItem();
        RayleighTable->setHorizontalHeaderItem(1, __qtablewidgetitem908);
        if (RayleighTable->rowCount() < 20)
            RayleighTable->setRowCount(20);
        QTableWidgetItem *__qtablewidgetitem909 = new QTableWidgetItem();
        RayleighTable->setVerticalHeaderItem(0, __qtablewidgetitem909);
        QTableWidgetItem *__qtablewidgetitem910 = new QTableWidgetItem();
        RayleighTable->setVerticalHeaderItem(1, __qtablewidgetitem910);
        QTableWidgetItem *__qtablewidgetitem911 = new QTableWidgetItem();
        RayleighTable->setVerticalHeaderItem(2, __qtablewidgetitem911);
        QTableWidgetItem *__qtablewidgetitem912 = new QTableWidgetItem();
        RayleighTable->setVerticalHeaderItem(3, __qtablewidgetitem912);
        QTableWidgetItem *__qtablewidgetitem913 = new QTableWidgetItem();
        RayleighTable->setVerticalHeaderItem(4, __qtablewidgetitem913);
        QTableWidgetItem *__qtablewidgetitem914 = new QTableWidgetItem();
        RayleighTable->setVerticalHeaderItem(5, __qtablewidgetitem914);
        QTableWidgetItem *__qtablewidgetitem915 = new QTableWidgetItem();
        RayleighTable->setVerticalHeaderItem(6, __qtablewidgetitem915);
        QTableWidgetItem *__qtablewidgetitem916 = new QTableWidgetItem();
        RayleighTable->setVerticalHeaderItem(7, __qtablewidgetitem916);
        QTableWidgetItem *__qtablewidgetitem917 = new QTableWidgetItem();
        RayleighTable->setVerticalHeaderItem(8, __qtablewidgetitem917);
        QTableWidgetItem *__qtablewidgetitem918 = new QTableWidgetItem();
        RayleighTable->setVerticalHeaderItem(9, __qtablewidgetitem918);
        QTableWidgetItem *__qtablewidgetitem919 = new QTableWidgetItem();
        RayleighTable->setVerticalHeaderItem(10, __qtablewidgetitem919);
        QTableWidgetItem *__qtablewidgetitem920 = new QTableWidgetItem();
        RayleighTable->setVerticalHeaderItem(11, __qtablewidgetitem920);
        QTableWidgetItem *__qtablewidgetitem921 = new QTableWidgetItem();
        RayleighTable->setVerticalHeaderItem(12, __qtablewidgetitem921);
        QTableWidgetItem *__qtablewidgetitem922 = new QTableWidgetItem();
        RayleighTable->setVerticalHeaderItem(13, __qtablewidgetitem922);
        QTableWidgetItem *__qtablewidgetitem923 = new QTableWidgetItem();
        RayleighTable->setVerticalHeaderItem(14, __qtablewidgetitem923);
        QTableWidgetItem *__qtablewidgetitem924 = new QTableWidgetItem();
        RayleighTable->setVerticalHeaderItem(15, __qtablewidgetitem924);
        QTableWidgetItem *__qtablewidgetitem925 = new QTableWidgetItem();
        RayleighTable->setVerticalHeaderItem(16, __qtablewidgetitem925);
        QTableWidgetItem *__qtablewidgetitem926 = new QTableWidgetItem();
        RayleighTable->setVerticalHeaderItem(17, __qtablewidgetitem926);
        QTableWidgetItem *__qtablewidgetitem927 = new QTableWidgetItem();
        RayleighTable->setVerticalHeaderItem(18, __qtablewidgetitem927);
        QTableWidgetItem *__qtablewidgetitem928 = new QTableWidgetItem();
        RayleighTable->setVerticalHeaderItem(19, __qtablewidgetitem928);
        RayleighTable->setObjectName(QString::fromUtf8("RayleighTable"));
        sizePolicy2.setHeightForWidth(RayleighTable->sizePolicy().hasHeightForWidth());
        RayleighTable->setSizePolicy(sizePolicy2);
        RayleighTable->setRowCount(20);
        RayleighTable->setColumnCount(2);

        verticalLayout_32->addWidget(RayleighTable);


        horizontalLayout_25->addWidget(RayleighGroupBox);


        verticalLayout_43->addLayout(horizontalLayout_25);

        TabWidgetRZ->addTab(MCTPRegTab, QString());
        PLOTtab = new QWidget();
        PLOTtab->setObjectName(QString::fromUtf8("PLOTtab"));
        gridLayout_22 = new QGridLayout(PLOTtab);
        gridLayout_22->setSpacing(6);
        gridLayout_22->setContentsMargins(11, 11, 11, 11);
        gridLayout_22->setObjectName(QString::fromUtf8("gridLayout_22"));
        verticalLayout_25 = new QVBoxLayout();
        verticalLayout_25->setSpacing(6);
        verticalLayout_25->setObjectName(QString::fromUtf8("verticalLayout_25"));
        plotCheckBox = new QCheckBox(PLOTtab);
        plotCheckBox->setObjectName(QString::fromUtf8("plotCheckBox"));

        verticalLayout_25->addWidget(plotCheckBox);

        PlotGroupBox = new QGroupBox(PLOTtab);
        PlotGroupBox->setObjectName(QString::fromUtf8("PlotGroupBox"));
        formLayout_2 = new QFormLayout(PlotGroupBox);
        formLayout_2->setSpacing(6);
        formLayout_2->setContentsMargins(11, 11, 11, 11);
        formLayout_2->setObjectName(QString::fromUtf8("formLayout_2"));
        horizontalLayout_13 = new QHBoxLayout();
        horizontalLayout_13->setSpacing(6);
        horizontalLayout_13->setObjectName(QString::fromUtf8("horizontalLayout_13"));
        verticalLayout_24 = new QVBoxLayout();
        verticalLayout_24->setSpacing(6);
        verticalLayout_24->setObjectName(QString::fromUtf8("verticalLayout_24"));
        verticalLayout_21 = new QVBoxLayout();
        verticalLayout_21->setSpacing(6);
        verticalLayout_21->setObjectName(QString::fromUtf8("verticalLayout_21"));
        LinePrnOutCheckBox = new QCheckBox(PlotGroupBox);
        LinePrnOutCheckBox->setObjectName(QString::fromUtf8("LinePrnOutCheckBox"));

        verticalLayout_21->addWidget(LinePrnOutCheckBox);

        ExtPlotOutCheckBox = new QCheckBox(PlotGroupBox);
        ExtPlotOutCheckBox->setObjectName(QString::fromUtf8("ExtPlotOutCheckBox"));

        verticalLayout_21->addWidget(ExtPlotOutCheckBox);


        verticalLayout_24->addLayout(verticalLayout_21);

        ExternalPlotTypeGroupBox = new QGroupBox(PlotGroupBox);
        ExternalPlotTypeGroupBox->setObjectName(QString::fromUtf8("ExternalPlotTypeGroupBox"));
        sizePolicy.setHeightForWidth(ExternalPlotTypeGroupBox->sizePolicy().hasHeightForWidth());
        ExternalPlotTypeGroupBox->setSizePolicy(sizePolicy);
        gridLayout_20 = new QGridLayout(ExternalPlotTypeGroupBox);
        gridLayout_20->setSpacing(6);
        gridLayout_20->setContentsMargins(11, 11, 11, 11);
        gridLayout_20->setObjectName(QString::fromUtf8("gridLayout_20"));
        ExternalPlotTypeComboBox = new QComboBox(ExternalPlotTypeGroupBox);
        ExternalPlotTypeComboBox->addItem(QString());
        ExternalPlotTypeComboBox->addItem(QString());
        ExternalPlotTypeComboBox->addItem(QString());
        ExternalPlotTypeComboBox->setObjectName(QString::fromUtf8("ExternalPlotTypeComboBox"));

        gridLayout_20->addWidget(ExternalPlotTypeComboBox, 0, 0, 1, 1);


        verticalLayout_24->addWidget(ExternalPlotTypeGroupBox);

        PlotRegionsGroupBox = new QGroupBox(PlotGroupBox);
        PlotRegionsGroupBox->setObjectName(QString::fromUtf8("PlotRegionsGroupBox"));
        PlotRegionsGroupBox->setMaximumSize(QSize(270, 16777215));
        horizontalLayout_23 = new QHBoxLayout(PlotRegionsGroupBox);
        horizontalLayout_23->setSpacing(6);
        horizontalLayout_23->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_23->setObjectName(QString::fromUtf8("horizontalLayout_23"));
        PlotRegionsTable = new QTableWidget(PlotRegionsGroupBox);
        if (PlotRegionsTable->columnCount() < 2)
            PlotRegionsTable->setColumnCount(2);
        QTableWidgetItem *__qtablewidgetitem929 = new QTableWidgetItem();
        PlotRegionsTable->setHorizontalHeaderItem(0, __qtablewidgetitem929);
        QTableWidgetItem *__qtablewidgetitem930 = new QTableWidgetItem();
        PlotRegionsTable->setHorizontalHeaderItem(1, __qtablewidgetitem930);
        if (PlotRegionsTable->rowCount() < 50)
            PlotRegionsTable->setRowCount(50);
        QTableWidgetItem *__qtablewidgetitem931 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(0, __qtablewidgetitem931);
        QTableWidgetItem *__qtablewidgetitem932 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(1, __qtablewidgetitem932);
        QTableWidgetItem *__qtablewidgetitem933 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(2, __qtablewidgetitem933);
        QTableWidgetItem *__qtablewidgetitem934 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(3, __qtablewidgetitem934);
        QTableWidgetItem *__qtablewidgetitem935 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(4, __qtablewidgetitem935);
        QTableWidgetItem *__qtablewidgetitem936 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(5, __qtablewidgetitem936);
        QTableWidgetItem *__qtablewidgetitem937 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(6, __qtablewidgetitem937);
        QTableWidgetItem *__qtablewidgetitem938 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(7, __qtablewidgetitem938);
        QTableWidgetItem *__qtablewidgetitem939 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(8, __qtablewidgetitem939);
        QTableWidgetItem *__qtablewidgetitem940 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(9, __qtablewidgetitem940);
        QTableWidgetItem *__qtablewidgetitem941 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(10, __qtablewidgetitem941);
        QTableWidgetItem *__qtablewidgetitem942 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(11, __qtablewidgetitem942);
        QTableWidgetItem *__qtablewidgetitem943 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(12, __qtablewidgetitem943);
        QTableWidgetItem *__qtablewidgetitem944 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(13, __qtablewidgetitem944);
        QTableWidgetItem *__qtablewidgetitem945 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(14, __qtablewidgetitem945);
        QTableWidgetItem *__qtablewidgetitem946 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(15, __qtablewidgetitem946);
        QTableWidgetItem *__qtablewidgetitem947 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(16, __qtablewidgetitem947);
        QTableWidgetItem *__qtablewidgetitem948 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(17, __qtablewidgetitem948);
        QTableWidgetItem *__qtablewidgetitem949 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(18, __qtablewidgetitem949);
        QTableWidgetItem *__qtablewidgetitem950 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(19, __qtablewidgetitem950);
        QTableWidgetItem *__qtablewidgetitem951 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(20, __qtablewidgetitem951);
        QTableWidgetItem *__qtablewidgetitem952 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(21, __qtablewidgetitem952);
        QTableWidgetItem *__qtablewidgetitem953 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(22, __qtablewidgetitem953);
        QTableWidgetItem *__qtablewidgetitem954 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(23, __qtablewidgetitem954);
        QTableWidgetItem *__qtablewidgetitem955 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(24, __qtablewidgetitem955);
        QTableWidgetItem *__qtablewidgetitem956 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(25, __qtablewidgetitem956);
        QTableWidgetItem *__qtablewidgetitem957 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(26, __qtablewidgetitem957);
        QTableWidgetItem *__qtablewidgetitem958 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(27, __qtablewidgetitem958);
        QTableWidgetItem *__qtablewidgetitem959 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(28, __qtablewidgetitem959);
        QTableWidgetItem *__qtablewidgetitem960 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(29, __qtablewidgetitem960);
        QTableWidgetItem *__qtablewidgetitem961 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(30, __qtablewidgetitem961);
        QTableWidgetItem *__qtablewidgetitem962 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(31, __qtablewidgetitem962);
        QTableWidgetItem *__qtablewidgetitem963 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(32, __qtablewidgetitem963);
        QTableWidgetItem *__qtablewidgetitem964 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(33, __qtablewidgetitem964);
        QTableWidgetItem *__qtablewidgetitem965 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(34, __qtablewidgetitem965);
        QTableWidgetItem *__qtablewidgetitem966 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(35, __qtablewidgetitem966);
        QTableWidgetItem *__qtablewidgetitem967 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(36, __qtablewidgetitem967);
        QTableWidgetItem *__qtablewidgetitem968 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(37, __qtablewidgetitem968);
        QTableWidgetItem *__qtablewidgetitem969 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(38, __qtablewidgetitem969);
        QTableWidgetItem *__qtablewidgetitem970 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(39, __qtablewidgetitem970);
        QTableWidgetItem *__qtablewidgetitem971 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(40, __qtablewidgetitem971);
        QTableWidgetItem *__qtablewidgetitem972 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(41, __qtablewidgetitem972);
        QTableWidgetItem *__qtablewidgetitem973 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(42, __qtablewidgetitem973);
        QTableWidgetItem *__qtablewidgetitem974 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(43, __qtablewidgetitem974);
        QTableWidgetItem *__qtablewidgetitem975 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(44, __qtablewidgetitem975);
        QTableWidgetItem *__qtablewidgetitem976 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(45, __qtablewidgetitem976);
        QTableWidgetItem *__qtablewidgetitem977 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(46, __qtablewidgetitem977);
        QTableWidgetItem *__qtablewidgetitem978 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(47, __qtablewidgetitem978);
        QTableWidgetItem *__qtablewidgetitem979 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(48, __qtablewidgetitem979);
        QTableWidgetItem *__qtablewidgetitem980 = new QTableWidgetItem();
        PlotRegionsTable->setVerticalHeaderItem(49, __qtablewidgetitem980);
        PlotRegionsTable->setObjectName(QString::fromUtf8("PlotRegionsTable"));
        sizePolicy2.setHeightForWidth(PlotRegionsTable->sizePolicy().hasHeightForWidth());
        PlotRegionsTable->setSizePolicy(sizePolicy2);
        PlotRegionsTable->setMinimumSize(QSize(0, 0));
        PlotRegionsTable->setMaximumSize(QSize(10000, 10000));
        PlotRegionsTable->setRowCount(50);
        PlotRegionsTable->setColumnCount(2);

        horizontalLayout_23->addWidget(PlotRegionsTable);


        verticalLayout_24->addWidget(PlotRegionsGroupBox);


        horizontalLayout_13->addLayout(verticalLayout_24);

        horizontalSpacer_4 = new QSpacerItem(208, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_13->addItem(horizontalSpacer_4);

        gridLayout_21 = new QGridLayout();
        gridLayout_21->setSpacing(6);
        gridLayout_21->setObjectName(QString::fromUtf8("gridLayout_21"));
        verticalLayout_22 = new QVBoxLayout();
        verticalLayout_22->setSpacing(6);
        verticalLayout_22->setObjectName(QString::fromUtf8("verticalLayout_22"));
        eminusPlotCheckBox = new QCheckBox(PlotGroupBox);
        eminusPlotCheckBox->setObjectName(QString::fromUtf8("eminusPlotCheckBox"));

        verticalLayout_22->addWidget(eminusPlotCheckBox);

        eplusPlotCheckBox = new QCheckBox(PlotGroupBox);
        eplusPlotCheckBox->setObjectName(QString::fromUtf8("eplusPlotCheckBox"));

        verticalLayout_22->addWidget(eplusPlotCheckBox);


        gridLayout_21->addLayout(verticalLayout_22, 0, 0, 1, 1);

        verticalLayout_23 = new QVBoxLayout();
        verticalLayout_23->setSpacing(6);
        verticalLayout_23->setObjectName(QString::fromUtf8("verticalLayout_23"));
        gammaPlotCheckBox = new QCheckBox(PlotGroupBox);
        gammaPlotCheckBox->setObjectName(QString::fromUtf8("gammaPlotCheckBox"));

        verticalLayout_23->addWidget(gammaPlotCheckBox);

        ePlotCheckBox = new QCheckBox(PlotGroupBox);
        ePlotCheckBox->setObjectName(QString::fromUtf8("ePlotCheckBox"));

        verticalLayout_23->addWidget(ePlotCheckBox);


        gridLayout_21->addLayout(verticalLayout_23, 0, 1, 1, 1);

        DrawFluPlotsGroupBox = new QGroupBox(PlotGroupBox);
        DrawFluPlotsGroupBox->setObjectName(QString::fromUtf8("DrawFluPlotsGroupBox"));
        sizePolicy.setHeightForWidth(DrawFluPlotsGroupBox->sizePolicy().hasHeightForWidth());
        DrawFluPlotsGroupBox->setSizePolicy(sizePolicy);
        gridLayout_19 = new QGridLayout(DrawFluPlotsGroupBox);
        gridLayout_19->setSpacing(6);
        gridLayout_19->setContentsMargins(11, 11, 11, 11);
        gridLayout_19->setObjectName(QString::fromUtf8("gridLayout_19"));
        DrawFluPlotsComboBox = new QComboBox(DrawFluPlotsGroupBox);
        DrawFluPlotsComboBox->addItem(QString());
        DrawFluPlotsComboBox->addItem(QString());
        DrawFluPlotsComboBox->addItem(QString());
        DrawFluPlotsComboBox->setObjectName(QString::fromUtf8("DrawFluPlotsComboBox"));

        gridLayout_19->addWidget(DrawFluPlotsComboBox, 0, 0, 1, 1);


        gridLayout_21->addWidget(DrawFluPlotsGroupBox, 1, 0, 1, 2);

        SpecPlotGroupBox = new QGroupBox(PlotGroupBox);
        SpecPlotGroupBox->setObjectName(QString::fromUtf8("SpecPlotGroupBox"));
        SpecPlotGroupBox->setMaximumSize(QSize(270, 16777215));
        horizontalLayout_14 = new QHBoxLayout(SpecPlotGroupBox);
        horizontalLayout_14->setSpacing(6);
        horizontalLayout_14->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_14->setObjectName(QString::fromUtf8("horizontalLayout_14"));
        SpecPlotTable = new QTableWidget(SpecPlotGroupBox);
        if (SpecPlotTable->columnCount() < 2)
            SpecPlotTable->setColumnCount(2);
        QTableWidgetItem *__qtablewidgetitem981 = new QTableWidgetItem();
        SpecPlotTable->setHorizontalHeaderItem(0, __qtablewidgetitem981);
        QTableWidgetItem *__qtablewidgetitem982 = new QTableWidgetItem();
        SpecPlotTable->setHorizontalHeaderItem(1, __qtablewidgetitem982);
        if (SpecPlotTable->rowCount() < 50)
            SpecPlotTable->setRowCount(50);
        QTableWidgetItem *__qtablewidgetitem983 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(0, __qtablewidgetitem983);
        QTableWidgetItem *__qtablewidgetitem984 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(1, __qtablewidgetitem984);
        QTableWidgetItem *__qtablewidgetitem985 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(2, __qtablewidgetitem985);
        QTableWidgetItem *__qtablewidgetitem986 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(3, __qtablewidgetitem986);
        QTableWidgetItem *__qtablewidgetitem987 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(4, __qtablewidgetitem987);
        QTableWidgetItem *__qtablewidgetitem988 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(5, __qtablewidgetitem988);
        QTableWidgetItem *__qtablewidgetitem989 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(6, __qtablewidgetitem989);
        QTableWidgetItem *__qtablewidgetitem990 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(7, __qtablewidgetitem990);
        QTableWidgetItem *__qtablewidgetitem991 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(8, __qtablewidgetitem991);
        QTableWidgetItem *__qtablewidgetitem992 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(9, __qtablewidgetitem992);
        QTableWidgetItem *__qtablewidgetitem993 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(10, __qtablewidgetitem993);
        QTableWidgetItem *__qtablewidgetitem994 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(11, __qtablewidgetitem994);
        QTableWidgetItem *__qtablewidgetitem995 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(12, __qtablewidgetitem995);
        QTableWidgetItem *__qtablewidgetitem996 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(13, __qtablewidgetitem996);
        QTableWidgetItem *__qtablewidgetitem997 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(14, __qtablewidgetitem997);
        QTableWidgetItem *__qtablewidgetitem998 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(15, __qtablewidgetitem998);
        QTableWidgetItem *__qtablewidgetitem999 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(16, __qtablewidgetitem999);
        QTableWidgetItem *__qtablewidgetitem1000 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(17, __qtablewidgetitem1000);
        QTableWidgetItem *__qtablewidgetitem1001 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(18, __qtablewidgetitem1001);
        QTableWidgetItem *__qtablewidgetitem1002 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(19, __qtablewidgetitem1002);
        QTableWidgetItem *__qtablewidgetitem1003 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(20, __qtablewidgetitem1003);
        QTableWidgetItem *__qtablewidgetitem1004 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(21, __qtablewidgetitem1004);
        QTableWidgetItem *__qtablewidgetitem1005 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(22, __qtablewidgetitem1005);
        QTableWidgetItem *__qtablewidgetitem1006 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(23, __qtablewidgetitem1006);
        QTableWidgetItem *__qtablewidgetitem1007 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(24, __qtablewidgetitem1007);
        QTableWidgetItem *__qtablewidgetitem1008 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(25, __qtablewidgetitem1008);
        QTableWidgetItem *__qtablewidgetitem1009 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(26, __qtablewidgetitem1009);
        QTableWidgetItem *__qtablewidgetitem1010 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(27, __qtablewidgetitem1010);
        QTableWidgetItem *__qtablewidgetitem1011 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(28, __qtablewidgetitem1011);
        QTableWidgetItem *__qtablewidgetitem1012 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(29, __qtablewidgetitem1012);
        QTableWidgetItem *__qtablewidgetitem1013 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(30, __qtablewidgetitem1013);
        QTableWidgetItem *__qtablewidgetitem1014 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(31, __qtablewidgetitem1014);
        QTableWidgetItem *__qtablewidgetitem1015 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(32, __qtablewidgetitem1015);
        QTableWidgetItem *__qtablewidgetitem1016 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(33, __qtablewidgetitem1016);
        QTableWidgetItem *__qtablewidgetitem1017 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(34, __qtablewidgetitem1017);
        QTableWidgetItem *__qtablewidgetitem1018 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(35, __qtablewidgetitem1018);
        QTableWidgetItem *__qtablewidgetitem1019 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(36, __qtablewidgetitem1019);
        QTableWidgetItem *__qtablewidgetitem1020 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(37, __qtablewidgetitem1020);
        QTableWidgetItem *__qtablewidgetitem1021 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(38, __qtablewidgetitem1021);
        QTableWidgetItem *__qtablewidgetitem1022 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(39, __qtablewidgetitem1022);
        QTableWidgetItem *__qtablewidgetitem1023 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(40, __qtablewidgetitem1023);
        QTableWidgetItem *__qtablewidgetitem1024 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(41, __qtablewidgetitem1024);
        QTableWidgetItem *__qtablewidgetitem1025 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(42, __qtablewidgetitem1025);
        QTableWidgetItem *__qtablewidgetitem1026 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(43, __qtablewidgetitem1026);
        QTableWidgetItem *__qtablewidgetitem1027 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(44, __qtablewidgetitem1027);
        QTableWidgetItem *__qtablewidgetitem1028 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(45, __qtablewidgetitem1028);
        QTableWidgetItem *__qtablewidgetitem1029 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(46, __qtablewidgetitem1029);
        QTableWidgetItem *__qtablewidgetitem1030 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(47, __qtablewidgetitem1030);
        QTableWidgetItem *__qtablewidgetitem1031 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(48, __qtablewidgetitem1031);
        QTableWidgetItem *__qtablewidgetitem1032 = new QTableWidgetItem();
        SpecPlotTable->setVerticalHeaderItem(49, __qtablewidgetitem1032);
        SpecPlotTable->setObjectName(QString::fromUtf8("SpecPlotTable"));
        sizePolicy2.setHeightForWidth(SpecPlotTable->sizePolicy().hasHeightForWidth());
        SpecPlotTable->setSizePolicy(sizePolicy2);
        SpecPlotTable->setMinimumSize(QSize(0, 0));
        SpecPlotTable->setMaximumSize(QSize(10000, 10000));
        SpecPlotTable->setRowCount(50);
        SpecPlotTable->setColumnCount(2);

        horizontalLayout_14->addWidget(SpecPlotTable);


        gridLayout_21->addWidget(SpecPlotGroupBox, 2, 0, 1, 2);


        horizontalLayout_13->addLayout(gridLayout_21);


        formLayout_2->setLayout(0, QFormLayout::LabelRole, horizontalLayout_13);


        verticalLayout_25->addWidget(PlotGroupBox);


        gridLayout_22->addLayout(verticalLayout_25, 0, 0, 1, 1);

        TabWidgetRZ->addTab(PLOTtab, QString());

        gridLayout_7->addWidget(TabWidgetRZ, 1, 0, 1, 1);

        QWidget::setTabOrder(TitleEdit, cavrzRadioButton);
        QWidget::setTabOrder(cavrzRadioButton, dosrzRadioButton);
        QWidget::setTabOrder(dosrzRadioButton, flurzRadioButton);
        QWidget::setTabOrder(flurzRadioButton, OptradioButton);
        QWidget::setTabOrder(OptradioButton, NoOptradioButton);
        QWidget::setTabOrder(NoOptradioButton, DebugradioButton);
        QWidget::setTabOrder(DebugradioButton, CleanradioButton);
        QWidget::setTabOrder(CleanradioButton, HOMERadioButton);
        QWidget::setTabOrder(HOMERadioButton, HEN_HOUSERadioButton);
        QWidget::setTabOrder(HEN_HOUSERadioButton, OtherAreaRadioButton);
        QWidget::setTabOrder(OtherAreaRadioButton, HOMEPegsRadioButton);
        QWidget::setTabOrder(HOMEPegsRadioButton, HEN_HOUSEPegsRadioButton);
        QWidget::setTabOrder(HEN_HOUSEPegsRadioButton, OtherPegsAreaRadioButton);
        QWidget::setTabOrder(OtherPegsAreaRadioButton, InputFileComboBox);
        QWidget::setTabOrder(InputFileComboBox, pegs4ComboBox);
        QWidget::setTabOrder(pegs4ComboBox, CONFcomboBox);
        QWidget::setTabOrder(CONFcomboBox, OpenFileButton);
        QWidget::setTabOrder(OpenFileButton, Pegs4FileButton);
        QWidget::setTabOrder(Pegs4FileButton, SPECButton);
        QWidget::setTabOrder(SPECButton, ConfigurationButton);
        QWidget::setTabOrder(ConfigurationButton, logfileButton);
        QWidget::setTabOrder(logfileButton, ExecuteButton);
        QWidget::setTabOrder(ExecuteButton, PreviewRZButton);
        QWidget::setTabOrder(PreviewRZButton, PrintButton);
        QWidget::setTabOrder(PrintButton, compileButton);
        QWidget::setTabOrder(compileButton, ApplyButton);
        QWidget::setTabOrder(ApplyButton, CancelButton);
        QWidget::setTabOrder(CancelButton, HelpButton);
        QWidget::setTabOrder(HelpButton, TabWidgetRZ);
        QWidget::setTabOrder(TabWidgetRZ, noRandRadioButton);
        QWidget::setTabOrder(noRandRadioButton, lastRandRadioButton);
        QWidget::setTabOrder(lastRandRadioButton, allRandRadioButton);
        QWidget::setTabOrder(allRandRadioButton, storeDataCheckBox);
        QWidget::setTabOrder(storeDataCheckBox, PrintFluSpeComboBox);
        QWidget::setTabOrder(PrintFluSpeComboBox, sloteFluEdit);
        QWidget::setTabOrder(sloteFluEdit, sproutComboBox);
        QWidget::setTabOrder(sproutComboBox, outoptComboBox);
        QWidget::setTabOrder(outoptComboBox, etransportComboBox);
        QWidget::setTabOrder(etransportComboBox, iwatchComboBox);
        QWidget::setTabOrder(iwatchComboBox, irestartComboBox);
        QWidget::setTabOrder(irestartComboBox, minPlnSpinBox);
        QWidget::setTabOrder(minPlnSpinBox, maxPlnSpinBox);
        QWidget::setTabOrder(maxPlnSpinBox, minCylSpinBox);
        QWidget::setTabOrder(minCylSpinBox, maxCylSpinBox);
        QWidget::setTabOrder(maxCylSpinBox, IPRIMARYComboBox);
        QWidget::setTabOrder(IPRIMARYComboBox, kermaCheckBox);
        QWidget::setTabOrder(kermaCheckBox, photregCheckBox);
        QWidget::setTabOrder(photregCheckBox, ncaseEdit);
        QWidget::setTabOrder(ncaseEdit, maxCPUEdit);
        QWidget::setTabOrder(maxCPUEdit, statEdit);
        QWidget::setTabOrder(statEdit, ifullComboBox);
        QWidget::setTabOrder(ifullComboBox, rand1SpinBox);
        QWidget::setTabOrder(rand1SpinBox, rand2SpinBox);
        QWidget::setTabOrder(rand2SpinBox, SLOTEEdit);
        QWidget::setTabOrder(SLOTEEdit, DELTAEEdit);
        QWidget::setTabOrder(DELTAEEdit, ZFaceEdit);
        QWidget::setTabOrder(ZFaceEdit, groupRadioButton);
        QWidget::setTabOrder(groupRadioButton, individualRadioButton);
        QWidget::setTabOrder(individualRadioButton, cavityRadioButton);
        QWidget::setTabOrder(cavityRadioButton, mediaComboBox);
        QWidget::setTabOrder(mediaComboBox, parallelplateRadioButton);
        QWidget::setTabOrder(parallelplateRadioButton, wallthickEdit);
        QWidget::setTabOrder(wallthickEdit, cavradEdit);
        QWidget::setTabOrder(cavradEdit, cavlenEdit);
        QWidget::setTabOrder(cavlenEdit, electradEdit);
        QWidget::setTabOrder(electradEdit, wallmaterialComboBox);
        QWidget::setTabOrder(wallmaterialComboBox, electrmatComboBox);
        QWidget::setTabOrder(electrmatComboBox, phasespacePushButton);
        QWidget::setTabOrder(phasespacePushButton, phasespaceComboBox);
        QWidget::setTabOrder(phasespaceComboBox, phRadioButton);
        QWidget::setTabOrder(phRadioButton, allRadioButton);
        QWidget::setTabOrder(allRadioButton, chargedRadioButton);
        QWidget::setTabOrder(chargedRadioButton, monoenergeticRadioButton);
        QWidget::setTabOrder(monoenergeticRadioButton, ini_energyEdit);
        QWidget::setTabOrder(ini_energyEdit, specfnameButton);
        QWidget::setTabOrder(specfnameButton, specfnameComboBox);
        QWidget::setTabOrder(specfnameComboBox, ioutspCheckBox);
        QWidget::setTabOrder(ioutspCheckBox, localRadioButton);
        QWidget::setTabOrder(localRadioButton, externalRadioButton);
        QWidget::setTabOrder(externalRadioButton, raddistfnameButton);
        QWidget::setTabOrder(raddistfnameButton, raddistfnameComboBox);
        QWidget::setTabOrder(raddistfnameComboBox, ioutrdistCheckBox);
        QWidget::setTabOrder(ioutrdistCheckBox, temp1Edit);
        QWidget::setTabOrder(temp1Edit, temp2Edit);
        QWidget::setTabOrder(temp2Edit, temp3Edit);
        QWidget::setTabOrder(temp3Edit, temp4Edit);
        QWidget::setTabOrder(temp4Edit, imodeComboBox);
        QWidget::setTabOrder(imodeComboBox, SpinCheckBox);
        QWidget::setTabOrder(SpinCheckBox, ESTEPEEdit);
        QWidget::setTabOrder(ESTEPEEdit, XImaxEdit);
        QWidget::setTabOrder(XImaxEdit, SkinDepthEdit);
        QWidget::setTabOrder(SkinDepthEdit, PairAngSamplingComboBox);
        QWidget::setTabOrder(PairAngSamplingComboBox, BremsAngSamplingComboBox);
        QWidget::setTabOrder(BremsAngSamplingComboBox, BremsXSectionComboBox);
        QWidget::setTabOrder(BremsXSectionComboBox, estep_algorithmComboBox);
        QWidget::setTabOrder(estep_algorithmComboBox, BCAComboBox);
        QWidget::setTabOrder(BCAComboBox, GPCUTEdit);
        QWidget::setTabOrder(GPCUTEdit, PCUTCheckBox);
        QWidget::setTabOrder(PCUTCheckBox, GECUTEdit);
        QWidget::setTabOrder(GECUTEdit, ECUTCheckBox);
        QWidget::setTabOrder(ECUTCheckBox, GSMAXEdit);
        QWidget::setTabOrder(GSMAXEdit, SMAXCheckBox);
        QWidget::setTabOrder(SMAXCheckBox, photonSplitSpinBox);
        QWidget::setTabOrder(photonSplitSpinBox, ESAVEINEdit);
        QWidget::setTabOrder(ESAVEINEdit, eRangeRejCheckBox);
        QWidget::setTabOrder(eRangeRejCheckBox, ChargedPartRRCheckBox);
        QWidget::setTabOrder(ChargedPartRRCheckBox, BremsSplitSpinBox);
        QWidget::setTabOrder(BremsSplitSpinBox, BremsSplitCheckBox);
        QWidget::setTabOrder(BremsSplitCheckBox, PhotonForcingCheckBox);
        QWidget::setTabOrder(PhotonForcingCheckBox, StartForcingSpinBox);
        QWidget::setTabOrder(StartForcingSpinBox, StopForcingSpinBox);
        QWidget::setTabOrder(StopForcingSpinBox, RRDepthEdit);
        QWidget::setTabOrder(RRDepthEdit, RRFractionEdit);
        QWidget::setTabOrder(RRFractionEdit, CSEnhancementSpinBox);
        QWidget::setTabOrder(CSEnhancementSpinBox, ExpTrafoCEdit);
        QWidget::setTabOrder(ExpTrafoCEdit, plotCheckBox);
        QWidget::setTabOrder(plotCheckBox, eplusPlotCheckBox);
        QWidget::setTabOrder(eplusPlotCheckBox, ePlotCheckBox);
        QWidget::setTabOrder(ePlotCheckBox, ExtPlotOutCheckBox);
        QWidget::setTabOrder(ExtPlotOutCheckBox, ExternalPlotTypeComboBox);
        QWidget::setTabOrder(ExternalPlotTypeComboBox, LinePrnOutCheckBox);
        QWidget::setTabOrder(LinePrnOutCheckBox, eminusPlotCheckBox);
        QWidget::setTabOrder(eminusPlotCheckBox, gammaPlotCheckBox);
        QWidget::setTabOrder(gammaPlotCheckBox, DrawFluPlotsComboBox);
        QWidget::setTabOrder(DrawFluPlotsComboBox, thimbleRadioButton);
        QWidget::setTabOrder(thimbleRadioButton, temp5Edit);
        QWidget::setTabOrder(temp5Edit, temp6Edit);
        QWidget::setTabOrder(temp6Edit, temp7Edit);

        retranslateUi(InputRZForm);
        QObject::connect(CancelButton, SIGNAL(clicked()), InputRZForm, SLOT(close()));
        QObject::connect(sproutComboBox, SIGNAL(activated(QString)), InputRZForm, SLOT(update_SprOutTableHeaders()));
        QObject::connect(sproutComboBox, SIGNAL(highlighted(QString)), InputRZForm, SLOT(update_SprOutTableHeaders()));
        QObject::connect(parallelplateRadioButton, SIGNAL(toggled(bool)), InputRZForm, SLOT(set_electr()));
        QObject::connect(localRadioButton, SIGNAL(toggled(bool)), InputRZForm, SLOT(set_local_external()));
        QObject::connect(externalRadioButton, SIGNAL(toggled(bool)), InputRZForm, SLOT(set_local_external()));
        QObject::connect(monoenergeticRadioButton, SIGNAL(toggled(bool)), InputRZForm, SLOT(set_mono_spectrum()));
        QObject::connect(spectrumRadioButton, SIGNAL(toggled(bool)), InputRZForm, SLOT(set_mono_spectrum()));
        QObject::connect(specfnameButton, SIGNAL(clicked()), InputRZForm, SLOT(GetSPECfile()));
        QObject::connect(phasespacePushButton, SIGNAL(clicked()), InputRZForm, SLOT(GetPHSPfile()));
        QObject::connect(raddistfnameButton, SIGNAL(clicked()), InputRZForm, SLOT(GetRDISTfile()));
        QObject::connect(PCUTCheckBox, SIGNAL(toggled(bool)), InputRZForm, SLOT(EnableTransportParamByRegions()));
        QObject::connect(ECUTCheckBox, SIGNAL(toggled(bool)), InputRZForm, SLOT(EnableTransportParamByRegions()));
        QObject::connect(SMAXCheckBox, SIGNAL(toggled(bool)), InputRZForm, SLOT(EnableTransportParamByRegions()));
        QObject::connect(PrintFluSpeComboBox, SIGNAL(highlighted(int)), InputRZForm, SLOT(activate_fluence_table()));
        QObject::connect(PrintFluSpeComboBox, SIGNAL(activated(int)), InputRZForm, SLOT(activate_fluence_table()));
        QObject::connect(BremsSplitCheckBox, SIGNAL(clicked()), InputRZForm, SLOT(Enable_BremsSplit()));
        QObject::connect(plotCheckBox, SIGNAL(clicked()), InputRZForm, SLOT(enable_plot()));
        QObject::connect(HOMERadioButton, SIGNAL(toggled(bool)), InputRZForm, SLOT(set_working_area()));
        QObject::connect(HEN_HOUSERadioButton, SIGNAL(toggled(bool)), InputRZForm, SLOT(set_working_area()));
        QObject::connect(OtherAreaRadioButton, SIGNAL(toggled(bool)), InputRZForm, SLOT(set_working_area()));
        QObject::connect(logfileButton, SIGNAL(clicked()), InputRZForm, SLOT(show_errors()));
        QObject::connect(HOMEPegsRadioButton, SIGNAL(toggled(bool)), InputRZForm, SLOT(set_data_area()));
        QObject::connect(HEN_HOUSEPegsRadioButton, SIGNAL(toggled(bool)), InputRZForm, SLOT(set_data_area()));
        QObject::connect(OtherPegsAreaRadioButton, SIGNAL(toggled(bool)), InputRZForm, SLOT(set_data_area()));
        QObject::connect(OpenFileButton, SIGNAL(clicked()), InputRZForm, SLOT(OpenEGSInpFile()));
        QObject::connect(Pegs4FileButton, SIGNAL(clicked()), InputRZForm, SLOT(GetPEGSfile()));
        QObject::connect(ApplyButton, SIGNAL(clicked()), InputRZForm, SLOT(UpDateInputRZFile()));
        QObject::connect(individualRadioButton, SIGNAL(clicked()), InputRZForm, SLOT(set_individual()));
        QObject::connect(groupRadioButton, SIGNAL(clicked()), InputRZForm, SLOT(set_group()));
        QObject::connect(cavityRadioButton, SIGNAL(toggled(bool)), InputRZForm, SLOT(set_cav_regions()));
        QObject::connect(cavityRadioButton, SIGNAL(toggled(bool)), InputRZForm, SLOT(set_cavity()));
        QObject::connect(HelpButton, SIGNAL(clicked()), InputRZForm, SLOT(show_help()));
        QObject::connect(eRangeRejCheckBox, SIGNAL(clicked()), InputRZForm, SLOT(update_range_rejection()));
        QObject::connect(PhotonForcingCheckBox, SIGNAL(clicked()), InputRZForm, SLOT(update_photon_forcing()));
        QObject::connect(ifullComboBox, SIGNAL(highlighted(QString)), InputRZForm, SLOT(activate_PulseHDistInputs()));
        QObject::connect(ifullComboBox, SIGNAL(activated(QString)), InputRZForm, SLOT(activate_PulseHDistInputs()));
        QObject::connect(InputFileComboBox, SIGNAL(activated(int)), InputRZForm, SLOT(change_input_file()));
        QObject::connect(cavrzRadioButton, SIGNAL(clicked()), InputRZForm, SLOT(update_usercode_open()));
        QObject::connect(sprrzRadioButton, SIGNAL(clicked()), InputRZForm, SLOT(update_usercode_open()));
        QObject::connect(dosrzRadioButton, SIGNAL(clicked()), InputRZForm, SLOT(update_usercode_open()));
        QObject::connect(flurzRadioButton, SIGNAL(clicked()), InputRZForm, SLOT(update_usercode_open()));
        QObject::connect(CleanradioButton, SIGNAL(toggled(bool)), InputRZForm, SLOT(cleanChecked(bool)));
        QObject::connect(OptradioButton, SIGNAL(toggled(bool)), InputRZForm, SLOT(cleanChecked(bool)));
        QObject::connect(NoOptradioButton, SIGNAL(toggled(bool)), InputRZForm, SLOT(cleanChecked(bool)));
        QObject::connect(DebugradioButton, SIGNAL(toggled(bool)), InputRZForm, SLOT(cleanChecked(bool)));
        QObject::connect(SPECButton, SIGNAL(clicked()), InputRZForm, SLOT(getCONFFile()));
        QObject::connect(ConfigurationButton, SIGNAL(clicked()), InputRZForm, SLOT(configure()));
        QObject::connect(CONFcomboBox, SIGNAL(activated(int)), InputRZForm, SLOT(change_config_file()));
        QObject::connect(AboutButton, SIGNAL(clicked()), InputRZForm, SLOT(show_about()));
        QObject::connect(BoundComptoncomboBox, SIGNAL(activated(QString)), InputRZForm, SLOT(EnableTransportParamByRegions()));
        QObject::connect(PEcomboBox, SIGNAL(activated(QString)), InputRZForm, SLOT(EnableTransportParamByRegions()));
        QObject::connect(RayleighcomboBox, SIGNAL(activated(QString)), InputRZForm, SLOT(EnableTransportParamByRegions()));
        QObject::connect(RelaxationcomboBox, SIGNAL(activated(QString)), InputRZForm, SLOT(EnableTransportParamByRegions()));
        QObject::connect(RayleighcomboBox, SIGNAL(activated(QString)), InputRZForm, SLOT(activate_ff_table()));
        QObject::connect(RayleighcomboBox, SIGNAL(highlighted(QString)), InputRZForm, SLOT(activate_ff_table()));
        QObject::connect(RayleighcomboBox, SIGNAL(highlighted(QString)), InputRZForm, SLOT(EnableTransportParamByRegions()));
        QObject::connect(RelaxationcomboBox, SIGNAL(highlighted(QString)), InputRZForm, SLOT(EnableTransportParamByRegions()));
        QObject::connect(PEcomboBox, SIGNAL(highlighted(QString)), InputRZForm, SLOT(EnableTransportParamByRegions()));
        QObject::connect(BoundComptoncomboBox, SIGNAL(highlighted(QString)), InputRZForm, SLOT(EnableTransportParamByRegions()));
        QObject::connect(PEGSlessRadioButton, SIGNAL(toggled(bool)), InputRZForm, SLOT(set_data_area()));
        QObject::connect(MDFileButton, SIGNAL(clicked()), InputRZForm, SLOT(GetMDfile()));
        QObject::connect(DFBrowse, SIGNAL(clicked()), InputRZForm, SLOT(GetDFfile()));
        QObject::connect(mediaComboBox, SIGNAL(activated(QString)), InputRZForm, SLOT(update_MediaInput()));
        QObject::connect(sourceComboBox, SIGNAL(activated(QString)), InputRZForm, SLOT(update_source_type()));
        QObject::connect(mediaTable, SIGNAL(cellDoubleClicked(int,int)), InputRZForm, SLOT(mediaTable_clicked(int,int)));
        QObject::connect(mediaTable, SIGNAL(cellClicked(int,int)), InputRZForm, SLOT(mediaTable_singleclicked(int,int)));
        QObject::connect(customFFTable, SIGNAL(cellDoubleClicked(int,int)), InputRZForm, SLOT(customFFTable_clicked(int,int)));
        QObject::connect(customFFTable, SIGNAL(cellClicked(int,int)), InputRZForm, SLOT(customFFTable_singleclicked(int,int)));
        QObject::connect(compileButton, SIGNAL(clicked()), InputRZForm, SLOT(compile_userCode()));
        QObject::connect(ExecuteButton, SIGNAL(clicked()), InputRZForm, SLOT(run_userCode()));
        QObject::connect(PreviewRZButton, SIGNAL(clicked()), InputRZForm, SLOT(run_previewRZ()));
        QObject::connect(ExtPlotOutCheckBox, SIGNAL(clicked()), InputRZForm, SLOT(enable_external_plot()));
        QObject::connect(isGasCheckBox, SIGNAL(clicked()), InputRZForm, SLOT(enable_gaspEdit()));
        QObject::connect(pz_or_rhozTable, SIGNAL(cellDoubleClicked(int,int)), InputRZForm, SLOT(pz_or_rhozTable_clicked(int,int)));
        QObject::connect(pz_or_rhozTable, SIGNAL(cellClicked(int,int)), InputRZForm, SLOT(pz_or_rhozTable_singleclicked(int,int)));
        QObject::connect(inpmediumComboBox, SIGNAL(highlighted(QString)), InputRZForm, SLOT(inpmediumSave(QString)));
        QObject::connect(inpmediumComboBox, SIGNAL(activated(QString)), InputRZForm, SLOT(inpmediumChanged(QString)));
        QObject::connect(DCcheckBox, SIGNAL(toggled(bool)), InputRZForm, SLOT(enableDCfileInput(bool)));
        QObject::connect(medTypeComboBox, SIGNAL(activated(QString)), InputRZForm, SLOT(medTypeChanged(QString)));
        QObject::connect(DFEdit, SIGNAL(returnPressed()), InputRZForm, SLOT(GetDFfileReturn()));

        TabWidgetRZ->setCurrentIndex(7);
        InputFileComboBox->setCurrentIndex(-1);


        QMetaObject::connectSlotsByName(InputRZForm);
    } // setupUi

    void retranslateUi(QWidget *InputRZForm)
    {
        InputRZForm->setWindowTitle(QCoreApplication::translate("InputRZForm", "GUI for RZ EGSnrc user codes. Copyright 2004 NRC Canada", nullptr));
        InputRZForm->setWindowIconText(QCoreApplication::translate("InputRZForm", "inpuRZ", nullptr));
#if QT_CONFIG(tooltip)
        InputRZForm->setToolTip(QString());
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        InputRZForm->setWhatsThis(QCoreApplication::translate("InputRZForm", "Nothing defined !", nullptr));
#endif // QT_CONFIG(whatsthis)
#if QT_CONFIG(tooltip)
        bannerLabel->setToolTip(QString());
#endif // QT_CONFIG(tooltip)
        ButtonGroup4->setTitle(QString());
#if QT_CONFIG(tooltip)
        ExecuteButton->setToolTip(QCoreApplication::translate("InputRZForm", "Only enabled if user code compiled, no input errors found and proper pegs data used.", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        ExecuteButton->setWhatsThis(QCoreApplication::translate("InputRZForm", "Only enabled if user code compiled, no input errors found and proper pegs data used.", nullptr));
#endif // QT_CONFIG(whatsthis)
        ExecuteButton->setText(QCoreApplication::translate("InputRZForm", "&Execute", nullptr));
#if QT_CONFIG(shortcut)
        ExecuteButton->setShortcut(QCoreApplication::translate("InputRZForm", "Alt+E", nullptr));
#endif // QT_CONFIG(shortcut)
#if QT_CONFIG(tooltip)
        PreviewRZButton->setToolTip(QCoreApplication::translate("InputRZForm", "preview geometry of an existing input file", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        PreviewRZButton->setWhatsThis(QCoreApplication::translate("InputRZForm", "preview geometry of an existing input file", nullptr));
#endif // QT_CONFIG(whatsthis)
        PreviewRZButton->setText(QCoreApplication::translate("InputRZForm", "Pre&viewRZ", nullptr));
#if QT_CONFIG(shortcut)
        PreviewRZButton->setShortcut(QCoreApplication::translate("InputRZForm", "Alt+V", nullptr));
#endif // QT_CONFIG(shortcut)
#if QT_CONFIG(tooltip)
        PrintButton->setToolTip(QCoreApplication::translate("InputRZForm", "print input file", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        PrintButton->setWhatsThis(QCoreApplication::translate("InputRZForm", "print input file", nullptr));
#endif // QT_CONFIG(whatsthis)
        PrintButton->setText(QCoreApplication::translate("InputRZForm", "&Print", nullptr));
#if QT_CONFIG(shortcut)
        PrintButton->setShortcut(QCoreApplication::translate("InputRZForm", "Alt+P", nullptr));
#endif // QT_CONFIG(shortcut)
#if QT_CONFIG(tooltip)
        compileButton->setToolTip(QCoreApplication::translate("InputRZForm", "Compile user code", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        compileButton->setWhatsThis(QCoreApplication::translate("InputRZForm", "Compile user code", nullptr));
#endif // QT_CONFIG(whatsthis)
        compileButton->setText(QCoreApplication::translate("InputRZForm", "&Compile", nullptr));
#if QT_CONFIG(shortcut)
        compileButton->setShortcut(QCoreApplication::translate("InputRZForm", "Alt+C", nullptr));
#endif // QT_CONFIG(shortcut)
        ConfirmButtonGroup->setTitle(QString());
#if QT_CONFIG(tooltip)
        ApplyButton->setToolTip(QCoreApplication::translate("InputRZForm", "save to/modify file !", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        ApplyButton->setWhatsThis(QCoreApplication::translate("InputRZForm", "save to/modify file !", nullptr));
#endif // QT_CONFIG(whatsthis)
        ApplyButton->setText(QCoreApplication::translate("InputRZForm", "&Save", nullptr));
#if QT_CONFIG(shortcut)
        ApplyButton->setShortcut(QCoreApplication::translate("InputRZForm", "Alt+S", nullptr));
#endif // QT_CONFIG(shortcut)
#if QT_CONFIG(tooltip)
        CancelButton->setToolTip(QCoreApplication::translate("InputRZForm", "close application without saving.", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        CancelButton->setWhatsThis(QCoreApplication::translate("InputRZForm", "close application without saving.", nullptr));
#endif // QT_CONFIG(whatsthis)
        CancelButton->setText(QCoreApplication::translate("InputRZForm", "&Exit", nullptr));
#if QT_CONFIG(shortcut)
        CancelButton->setShortcut(QCoreApplication::translate("InputRZForm", "Alt+E", nullptr));
#endif // QT_CONFIG(shortcut)
#if QT_CONFIG(tooltip)
        HelpButton->setToolTip(QCoreApplication::translate("InputRZForm", "shows the user manual with Netscape or Konqueror", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        HelpButton->setWhatsThis(QCoreApplication::translate("InputRZForm", "shows the user manual with Netscape or Konqueror", nullptr));
#endif // QT_CONFIG(whatsthis)
        HelpButton->setText(QCoreApplication::translate("InputRZForm", "&Help", nullptr));
#if QT_CONFIG(shortcut)
        HelpButton->setShortcut(QCoreApplication::translate("InputRZForm", "Alt+H", nullptr));
#endif // QT_CONFIG(shortcut)
        AboutButton->setText(QCoreApplication::translate("InputRZForm", "&About", nullptr));
#if QT_CONFIG(shortcut)
        AboutButton->setShortcut(QCoreApplication::translate("InputRZForm", "Alt+A", nullptr));
#endif // QT_CONFIG(shortcut)
#if QT_CONFIG(tooltip)
        UserCodeButtonGroup->setToolTip(QString());
#endif // QT_CONFIG(tooltip)
        UserCodeButtonGroup->setTitle(QCoreApplication::translate("InputRZForm", "EGSnrc user code", nullptr));
#if QT_CONFIG(tooltip)
        cavrzRadioButton->setToolTip(QCoreApplication::translate("InputRZForm", "calculates quantities of interest for right cylindrical ion chambers", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        cavrzRadioButton->setWhatsThis(QCoreApplication::translate("InputRZForm", "calculates quantities of interest for right cylindrical ion chambers", nullptr));
#endif // QT_CONFIG(whatsthis)
        cavrzRadioButton->setText(QCoreApplication::translate("InputRZForm", "CAVRZnrc", nullptr));
#if QT_CONFIG(tooltip)
        dosrzRadioButton->setToolTip(QCoreApplication::translate("InputRZForm", "scores dose deposited by a photon or electron beam in a right cylindrical geometry", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        dosrzRadioButton->setWhatsThis(QCoreApplication::translate("InputRZForm", "scores dose deposited by a photon or electron beam in a right cylindrical geometry", nullptr));
#endif // QT_CONFIG(whatsthis)
        dosrzRadioButton->setText(QCoreApplication::translate("InputRZForm", "DOSRZnrc", nullptr));
#if QT_CONFIG(tooltip)
        sprrzRadioButton->setToolTip(QCoreApplication::translate("InputRZForm", "calculates medium to air stopping power ratios in right cylindrical geometries", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        sprrzRadioButton->setWhatsThis(QCoreApplication::translate("InputRZForm", "calculates medium to air stopping power ratios in right cylindrical geometries", nullptr));
#endif // QT_CONFIG(whatsthis)
        sprrzRadioButton->setText(QCoreApplication::translate("InputRZForm", "SPRRZnrc", nullptr));
#if QT_CONFIG(tooltip)
        flurzRadioButton->setToolTip(QCoreApplication::translate("InputRZForm", "fluence spectra scored by summing track lengths in a zone", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        flurzRadioButton->setWhatsThis(QCoreApplication::translate("InputRZForm", "fluence spectra scored by summing track lengths in a zone", nullptr));
#endif // QT_CONFIG(whatsthis)
        flurzRadioButton->setText(QCoreApplication::translate("InputRZForm", "FLURZnrc", nullptr));
        TargetbuttonGroup->setTitle(QCoreApplication::translate("InputRZForm", "Target", nullptr));
        OptradioButton->setText(QCoreApplication::translate("InputRZForm", "optimization", nullptr));
        NoOptradioButton->setText(QCoreApplication::translate("InputRZForm", "no optimization", nullptr));
        DebugradioButton->setText(QCoreApplication::translate("InputRZForm", "debug", nullptr));
        CleanradioButton->setText(QCoreApplication::translate("InputRZForm", "clean", nullptr));
#if QT_CONFIG(tooltip)
        UserCodeAreaButtonGroup->setToolTip(QCoreApplication::translate("InputRZForm", "Select default area for user code and pegs4 data location", nullptr));
#endif // QT_CONFIG(tooltip)
        UserCodeAreaButtonGroup->setTitle(QCoreApplication::translate("InputRZForm", "User code area", nullptr));
        HOMERadioButton->setText(QCoreApplication::translate("InputRZForm", "EGS_HOME", nullptr));
        HEN_HOUSERadioButton->setText(QCoreApplication::translate("InputRZForm", "HEN_HOUSE", nullptr));
        OtherAreaRadioButton->setText(QCoreApplication::translate("InputRZForm", "Other", nullptr));
        PEGSDataAreaButtonGroup->setTitle(QCoreApplication::translate("InputRZForm", "Pegs data area", nullptr));
        HOMEPegsRadioButton->setText(QCoreApplication::translate("InputRZForm", "EGS_HOME", nullptr));
        HEN_HOUSEPegsRadioButton->setText(QCoreApplication::translate("InputRZForm", "HEN_HOUSE", nullptr));
        OtherPegsAreaRadioButton->setText(QCoreApplication::translate("InputRZForm", "Other", nullptr));
        PEGSlessRadioButton->setText(QCoreApplication::translate("InputRZForm", "Go PEGSless", nullptr));
#if QT_CONFIG(tooltip)
        TitleGroupBox->setToolTip(QCoreApplication::translate("InputRZForm", "title is maximum 80 characters wide", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        TitleGroupBox->setWhatsThis(QCoreApplication::translate("InputRZForm", "title is maximum 80 characters wide", nullptr));
#endif // QT_CONFIG(whatsthis)
        TitleGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Title (80 characters maximum)", nullptr));
#if QT_CONFIG(tooltip)
        TitleEdit->setToolTip(QCoreApplication::translate("InputRZForm", "title is maximum 80 characters wide", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        TitleEdit->setWhatsThis(QCoreApplication::translate("InputRZForm", "title is maximum 80 characters wide", nullptr));
#endif // QT_CONFIG(whatsthis)
        TitleEdit->setText(QCoreApplication::translate("InputRZForm", "EGSnrc input template file: Please modify this !!!", nullptr));
#if QT_CONFIG(tooltip)
        InputFileGroupBox->setToolTip(QCoreApplication::translate("InputRZForm", "select input file in current user code area.", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        InputFileGroupBox->setWhatsThis(QCoreApplication::translate("InputRZForm", "select input file in current user code area.", nullptr));
#endif // QT_CONFIG(whatsthis)
        InputFileGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "EGSnrc input file name (*.egsinp)", nullptr));
#if QT_CONFIG(tooltip)
        InputFileComboBox->setToolTip(QCoreApplication::translate("InputRZForm", "select input file in current user code area.", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        InputFileComboBox->setWhatsThis(QCoreApplication::translate("InputRZForm", "select input file in current user code area.", nullptr));
#endif // QT_CONFIG(whatsthis)
#if QT_CONFIG(tooltip)
        OpenFileButton->setToolTip(QCoreApplication::translate("InputRZForm", "load EGSnrc input file using an Open File Dialog.", nullptr));
#endif // QT_CONFIG(tooltip)
        OpenFileButton->setText(QCoreApplication::translate("InputRZForm", "...", nullptr));
#if QT_CONFIG(tooltip)
        pegs4GroupBox->setToolTip(QCoreApplication::translate("InputRZForm", "select pegs4 file in current pegs data area.", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        pegs4GroupBox->setWhatsThis(QCoreApplication::translate("InputRZForm", "select pegs4 file in current pegs data area.", nullptr));
#endif // QT_CONFIG(whatsthis)
        pegs4GroupBox->setTitle(QCoreApplication::translate("InputRZForm", "PEGS4 file name (*.pegs4dat)", nullptr));
#if QT_CONFIG(tooltip)
        pegs4ComboBox->setToolTip(QCoreApplication::translate("InputRZForm", "select pegs4 file in current pegs data area.", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        pegs4ComboBox->setWhatsThis(QCoreApplication::translate("InputRZForm", "select pegs4 file in current pegs data area.", nullptr));
#endif // QT_CONFIG(whatsthis)
#if QT_CONFIG(tooltip)
        Pegs4FileButton->setToolTip(QCoreApplication::translate("InputRZForm", "load pegs4 file using an Open File Dialog.", nullptr));
#endif // QT_CONFIG(tooltip)
        Pegs4FileButton->setText(QCoreApplication::translate("InputRZForm", "...", nullptr));
        groupBox57->setTitle(QCoreApplication::translate("InputRZForm", "Configuration file", nullptr));
        SPECButton->setText(QCoreApplication::translate("InputRZForm", "...", nullptr));
        ConfigurationButton->setText(QCoreApplication::translate("InputRZForm", "Configuration", nullptr));
#if QT_CONFIG(tooltip)
        logfileButton->setToolTip(QCoreApplication::translate("InputRZForm", "enabled if errors detected reading the file", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        logfileButton->setWhatsThis(QCoreApplication::translate("InputRZForm", "enabled if errors detected reading the file", nullptr));
#endif // QT_CONFIG(whatsthis)
        logfileButton->setText(QCoreApplication::translate("InputRZForm", "view errors", nullptr));
        TabWidgetRZ->setTabText(TabWidgetRZ->indexOf(GItab), QCoreApplication::translate("InputRZForm", "General", nullptr));
#if QT_CONFIG(tooltip)
        randNButtonGroup->setToolTip(QCoreApplication::translate("InputRZForm", "choose for which history initial random numbers should be stored", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        randNButtonGroup->setWhatsThis(QCoreApplication::translate("InputRZForm", "choose for which history initial random numbers should be stored", nullptr));
#endif // QT_CONFIG(whatsthis)
        randNButtonGroup->setTitle(QCoreApplication::translate("InputRZForm", "Store initial random numbers", nullptr));
#if QT_CONFIG(tooltip)
        noRandRadioButton->setToolTip(QCoreApplication::translate("InputRZForm", "don't store random numbers", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        noRandRadioButton->setWhatsThis(QCoreApplication::translate("InputRZForm", "don't store random numbers", nullptr));
#endif // QT_CONFIG(whatsthis)
        noRandRadioButton->setText(QCoreApplication::translate("InputRZForm", "no", nullptr));
#if QT_CONFIG(tooltip)
        lastRandRadioButton->setToolTip(QCoreApplication::translate("InputRZForm", "store initial random number for last history", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        lastRandRadioButton->setWhatsThis(QCoreApplication::translate("InputRZForm", "store initial random number for last history", nullptr));
#endif // QT_CONFIG(whatsthis)
        lastRandRadioButton->setText(QCoreApplication::translate("InputRZForm", "last", nullptr));
#if QT_CONFIG(tooltip)
        allRandRadioButton->setToolTip(QCoreApplication::translate("InputRZForm", "store initial random number for each history", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        allRandRadioButton->setWhatsThis(QCoreApplication::translate("InputRZForm", "store initial random number for each history", nullptr));
#endif // QT_CONFIG(whatsthis)
        allRandRadioButton->setText(QCoreApplication::translate("InputRZForm", "all", nullptr));
#if QT_CONFIG(tooltip)
        DoseRegGroupBox->setToolTip(QCoreApplication::translate("InputRZForm", "Define dose scoring regions", nullptr));
#endif // QT_CONFIG(tooltip)
        DoseRegGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Dose regions", nullptr));
#if QT_CONFIG(tooltip)
        minPlnTextLabel->setToolTip(QCoreApplication::translate("InputRZForm", "Minimum plane defining dose region", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        minPlnTextLabel->setWhatsThis(QCoreApplication::translate("InputRZForm", "Minimum plane defining dose region", nullptr));
#endif // QT_CONFIG(whatsthis)
        minPlnTextLabel->setText(QCoreApplication::translate("InputRZForm", "minimum plane", nullptr));
#if QT_CONFIG(tooltip)
        minPlnSpinBox->setToolTip(QCoreApplication::translate("InputRZForm", "Minimum plane defining dose region", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        minPlnSpinBox->setWhatsThis(QCoreApplication::translate("InputRZForm", "Minimum plane defining dose region", nullptr));
#endif // QT_CONFIG(whatsthis)
#if QT_CONFIG(tooltip)
        maxPlnTextLabel->setToolTip(QCoreApplication::translate("InputRZForm", "Maximum plane defining dose region", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        maxPlnTextLabel->setWhatsThis(QCoreApplication::translate("InputRZForm", "Maximum plane defining dose region", nullptr));
#endif // QT_CONFIG(whatsthis)
        maxPlnTextLabel->setText(QCoreApplication::translate("InputRZForm", "maximum plane", nullptr));
#if QT_CONFIG(tooltip)
        maxPlnSpinBox->setToolTip(QCoreApplication::translate("InputRZForm", "Maximum plane defining dose region", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        maxPlnSpinBox->setWhatsThis(QCoreApplication::translate("InputRZForm", "Maximum plane defining dose region", nullptr));
#endif // QT_CONFIG(whatsthis)
#if QT_CONFIG(tooltip)
        minCylTextLabel->setToolTip(QCoreApplication::translate("InputRZForm", "Minimum cylinder defining dose region", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        minCylTextLabel->setWhatsThis(QCoreApplication::translate("InputRZForm", "Minimum cylinder defining dose region", nullptr));
#endif // QT_CONFIG(whatsthis)
        minCylTextLabel->setText(QCoreApplication::translate("InputRZForm", "minimum cylinder", nullptr));
#if QT_CONFIG(tooltip)
        minCylSpinBox->setToolTip(QCoreApplication::translate("InputRZForm", "Minimum cylinder defining dose region", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        minCylSpinBox->setWhatsThis(QCoreApplication::translate("InputRZForm", "Minimum cylinder defining dose region", nullptr));
#endif // QT_CONFIG(whatsthis)
#if QT_CONFIG(tooltip)
        maxCylTextLabel->setToolTip(QCoreApplication::translate("InputRZForm", "Maximum cylinder defining dose region", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        maxCylTextLabel->setWhatsThis(QCoreApplication::translate("InputRZForm", "Maximum cylinder defining dose region", nullptr));
#endif // QT_CONFIG(whatsthis)
        maxCylTextLabel->setText(QCoreApplication::translate("InputRZForm", "maximum cylinder", nullptr));
#if QT_CONFIG(tooltip)
        maxCylSpinBox->setToolTip(QCoreApplication::translate("InputRZForm", "Maximum cylinder defining dose region", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        maxCylSpinBox->setWhatsThis(QCoreApplication::translate("InputRZForm", "Maximum cylinder defining dose region", nullptr));
#endif // QT_CONFIG(whatsthis)
#if QT_CONFIG(tooltip)
        Frame4->setToolTip(QCoreApplication::translate("InputRZForm", "output *.egsdat file for restarts, parallel post-processing, etc", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        Frame4->setWhatsThis(QCoreApplication::translate("InputRZForm", "output *.egsdat file for restarts, parallel post-processing, etc", nullptr));
#endif // QT_CONFIG(whatsthis)
#if QT_CONFIG(tooltip)
        storeDataCheckBox->setToolTip(QCoreApplication::translate("InputRZForm", "output *.egsdat file for restarts, parallel post-processing, etc", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        storeDataCheckBox->setWhatsThis(QCoreApplication::translate("InputRZForm", "output *.egsdat file for restarts, parallel post-processing, etc", nullptr));
#endif // QT_CONFIG(whatsthis)
        storeDataCheckBox->setText(QCoreApplication::translate("InputRZForm", "store data array", nullptr));
#if QT_CONFIG(tooltip)
        outoptTextLabel->setToolTip(QCoreApplication::translate("InputRZForm", "Select output type", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        outoptTextLabel->setWhatsThis(QCoreApplication::translate("InputRZForm", "Select output type", nullptr));
#endif // QT_CONFIG(whatsthis)
        outoptTextLabel->setText(QCoreApplication::translate("InputRZForm", "output options :", nullptr));
        outoptComboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "short", nullptr));
        outoptComboBox->setItemText(1, QCoreApplication::translate("InputRZForm", "cavity details", nullptr));
        outoptComboBox->setItemText(2, QCoreApplication::translate("InputRZForm", "material and dose summary", nullptr));

#if QT_CONFIG(tooltip)
        outoptComboBox->setToolTip(QCoreApplication::translate("InputRZForm", "Select output type", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        outoptComboBox->setWhatsThis(QCoreApplication::translate("InputRZForm", "Select output type", nullptr));
#endif // QT_CONFIG(whatsthis)
#if QT_CONFIG(tooltip)
        etransportLabel->setToolTip(QCoreApplication::translate("InputRZForm", "Define type of electron transport. See options for more detail.", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        etransportLabel->setWhatsThis(QCoreApplication::translate("InputRZForm", "Define type of electron transport. See options for more detail.", nullptr));
#endif // QT_CONFIG(whatsthis)
        etransportLabel->setText(QCoreApplication::translate("InputRZForm", "electron transport :", nullptr));
        etransportComboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "normal", nullptr));
        etransportComboBox->setItemText(1, QCoreApplication::translate("InputRZForm", "no interactions", nullptr));

#if QT_CONFIG(tooltip)
        etransportComboBox->setToolTip(QCoreApplication::translate("InputRZForm", "Define type of electron transport. See options for more detail.", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        etransportComboBox->setWhatsThis(QCoreApplication::translate("InputRZForm", "Define type of electron transport. See options for more detail.", nullptr));
#endif // QT_CONFIG(whatsthis)
#if QT_CONFIG(tooltip)
        iwatchTextLabel->setToolTip(QCoreApplication::translate("InputRZForm", "debug output with increasing detail, if not \"off\" use very few histories", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        iwatchTextLabel->setWhatsThis(QCoreApplication::translate("InputRZForm", "debug output with increasing detail, if not \"off\" use very few histories", nullptr));
#endif // QT_CONFIG(whatsthis)
        iwatchTextLabel->setText(QCoreApplication::translate("InputRZForm", "iwatch :", nullptr));
        iwatchComboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "off", nullptr));
        iwatchComboBox->setItemText(1, QCoreApplication::translate("InputRZForm", "interactions", nullptr));
        iwatchComboBox->setItemText(2, QCoreApplication::translate("InputRZForm", "steps", nullptr));
        iwatchComboBox->setItemText(3, QCoreApplication::translate("InputRZForm", "deposited", nullptr));
        iwatchComboBox->setItemText(4, QCoreApplication::translate("InputRZForm", "graph", nullptr));

#if QT_CONFIG(tooltip)
        iwatchComboBox->setToolTip(QCoreApplication::translate("InputRZForm", "Debug output with increasing detail, if not \"off\" use very few histories", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        iwatchComboBox->setWhatsThis(QCoreApplication::translate("InputRZForm", "Debug output with increasing detail, if not \"off\" use very few histories", nullptr));
#endif // QT_CONFIG(whatsthis)
#if QT_CONFIG(tooltip)
        restartTextLabel->setToolTip(QCoreApplication::translate("InputRZForm", "Define run mode", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        restartTextLabel->setWhatsThis(QCoreApplication::translate("InputRZForm", "Define run mode", nullptr));
#endif // QT_CONFIG(whatsthis)
        restartTextLabel->setText(QCoreApplication::translate("InputRZForm", "irestart :", nullptr));
        irestartComboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "first", nullptr));
        irestartComboBox->setItemText(1, QCoreApplication::translate("InputRZForm", "restart", nullptr));
        irestartComboBox->setItemText(2, QCoreApplication::translate("InputRZForm", "analyze", nullptr));
        irestartComboBox->setItemText(3, QCoreApplication::translate("InputRZForm", "start-RNS", nullptr));
        irestartComboBox->setItemText(4, QCoreApplication::translate("InputRZForm", "parallel", nullptr));

#if QT_CONFIG(tooltip)
        irestartComboBox->setToolTip(QCoreApplication::translate("InputRZForm", "run mode", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        irestartComboBox->setWhatsThis(QCoreApplication::translate("InputRZForm", "Run mode", nullptr));
#endif // QT_CONFIG(whatsthis)
        SPRRegGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Stopping power output regions", nullptr));
        sproutComboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "regions", nullptr));
        sproutComboBox->setItemText(1, QCoreApplication::translate("InputRZForm", "slabs/cylinders", nullptr));

#if QT_CONFIG(tooltip)
        sproutComboBox->setToolTip(QCoreApplication::translate("InputRZForm", "define output mode for stopping power ratios", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        sproutComboBox->setWhatsThis(QCoreApplication::translate("InputRZForm", "define output mode for stopping power ratios", nullptr));
#endif // QT_CONFIG(whatsthis)
#if QT_CONFIG(tooltip)
        sproutTextLabel->setToolTip(QCoreApplication::translate("InputRZForm", "define output mode for stopping power ratios", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        sproutTextLabel->setWhatsThis(QCoreApplication::translate("InputRZForm", "define output mode for stopping power ratios", nullptr));
#endif // QT_CONFIG(whatsthis)
        sproutTextLabel->setText(QCoreApplication::translate("InputRZForm", "spr output mode :", nullptr));
        QTableWidgetItem *___qtablewidgetitem = sproutTable->horizontalHeaderItem(0);
        ___qtablewidgetitem->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem1 = sproutTable->horizontalHeaderItem(1);
        ___qtablewidgetitem1->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem2 = sproutTable->verticalHeaderItem(0);
        ___qtablewidgetitem2->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem3 = sproutTable->verticalHeaderItem(1);
        ___qtablewidgetitem3->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem4 = sproutTable->verticalHeaderItem(2);
        ___qtablewidgetitem4->setText(QCoreApplication::translate("InputRZForm", "3", nullptr));
        QTableWidgetItem *___qtablewidgetitem5 = sproutTable->verticalHeaderItem(3);
        ___qtablewidgetitem5->setText(QCoreApplication::translate("InputRZForm", "4", nullptr));
        QTableWidgetItem *___qtablewidgetitem6 = sproutTable->verticalHeaderItem(4);
        ___qtablewidgetitem6->setText(QCoreApplication::translate("InputRZForm", "5", nullptr));
        QTableWidgetItem *___qtablewidgetitem7 = sproutTable->verticalHeaderItem(5);
        ___qtablewidgetitem7->setText(QCoreApplication::translate("InputRZForm", "6", nullptr));
        QTableWidgetItem *___qtablewidgetitem8 = sproutTable->verticalHeaderItem(6);
        ___qtablewidgetitem8->setText(QCoreApplication::translate("InputRZForm", "7", nullptr));
        QTableWidgetItem *___qtablewidgetitem9 = sproutTable->verticalHeaderItem(7);
        ___qtablewidgetitem9->setText(QCoreApplication::translate("InputRZForm", "8", nullptr));
        QTableWidgetItem *___qtablewidgetitem10 = sproutTable->verticalHeaderItem(8);
        ___qtablewidgetitem10->setText(QCoreApplication::translate("InputRZForm", "9", nullptr));
        QTableWidgetItem *___qtablewidgetitem11 = sproutTable->verticalHeaderItem(9);
        ___qtablewidgetitem11->setText(QCoreApplication::translate("InputRZForm", "10", nullptr));
        QTableWidgetItem *___qtablewidgetitem12 = sproutTable->verticalHeaderItem(10);
        ___qtablewidgetitem12->setText(QCoreApplication::translate("InputRZForm", "11", nullptr));
        QTableWidgetItem *___qtablewidgetitem13 = sproutTable->verticalHeaderItem(11);
        ___qtablewidgetitem13->setText(QCoreApplication::translate("InputRZForm", "12", nullptr));
        QTableWidgetItem *___qtablewidgetitem14 = sproutTable->verticalHeaderItem(12);
        ___qtablewidgetitem14->setText(QCoreApplication::translate("InputRZForm", "13", nullptr));
        QTableWidgetItem *___qtablewidgetitem15 = sproutTable->verticalHeaderItem(13);
        ___qtablewidgetitem15->setText(QCoreApplication::translate("InputRZForm", "14", nullptr));
        QTableWidgetItem *___qtablewidgetitem16 = sproutTable->verticalHeaderItem(14);
        ___qtablewidgetitem16->setText(QCoreApplication::translate("InputRZForm", "15", nullptr));
        QTableWidgetItem *___qtablewidgetitem17 = sproutTable->verticalHeaderItem(15);
        ___qtablewidgetitem17->setText(QCoreApplication::translate("InputRZForm", "16", nullptr));
        QTableWidgetItem *___qtablewidgetitem18 = sproutTable->verticalHeaderItem(16);
        ___qtablewidgetitem18->setText(QCoreApplication::translate("InputRZForm", "17", nullptr));
        QTableWidgetItem *___qtablewidgetitem19 = sproutTable->verticalHeaderItem(17);
        ___qtablewidgetitem19->setText(QCoreApplication::translate("InputRZForm", "18", nullptr));
        QTableWidgetItem *___qtablewidgetitem20 = sproutTable->verticalHeaderItem(18);
        ___qtablewidgetitem20->setText(QCoreApplication::translate("InputRZForm", "19", nullptr));
        QTableWidgetItem *___qtablewidgetitem21 = sproutTable->verticalHeaderItem(19);
        ___qtablewidgetitem21->setText(QCoreApplication::translate("InputRZForm", "20", nullptr));
        QTableWidgetItem *___qtablewidgetitem22 = sproutTable->verticalHeaderItem(20);
        ___qtablewidgetitem22->setText(QCoreApplication::translate("InputRZForm", "21", nullptr));
        QTableWidgetItem *___qtablewidgetitem23 = sproutTable->verticalHeaderItem(21);
        ___qtablewidgetitem23->setText(QCoreApplication::translate("InputRZForm", "22", nullptr));
        QTableWidgetItem *___qtablewidgetitem24 = sproutTable->verticalHeaderItem(22);
        ___qtablewidgetitem24->setText(QCoreApplication::translate("InputRZForm", "23", nullptr));
        QTableWidgetItem *___qtablewidgetitem25 = sproutTable->verticalHeaderItem(23);
        ___qtablewidgetitem25->setText(QCoreApplication::translate("InputRZForm", "24", nullptr));
        QTableWidgetItem *___qtablewidgetitem26 = sproutTable->verticalHeaderItem(24);
        ___qtablewidgetitem26->setText(QCoreApplication::translate("InputRZForm", "25", nullptr));
        QTableWidgetItem *___qtablewidgetitem27 = sproutTable->verticalHeaderItem(25);
        ___qtablewidgetitem27->setText(QCoreApplication::translate("InputRZForm", "26", nullptr));
        QTableWidgetItem *___qtablewidgetitem28 = sproutTable->verticalHeaderItem(26);
        ___qtablewidgetitem28->setText(QCoreApplication::translate("InputRZForm", "27", nullptr));
        QTableWidgetItem *___qtablewidgetitem29 = sproutTable->verticalHeaderItem(27);
        ___qtablewidgetitem29->setText(QCoreApplication::translate("InputRZForm", "28", nullptr));
        QTableWidgetItem *___qtablewidgetitem30 = sproutTable->verticalHeaderItem(28);
        ___qtablewidgetitem30->setText(QCoreApplication::translate("InputRZForm", "29", nullptr));
        QTableWidgetItem *___qtablewidgetitem31 = sproutTable->verticalHeaderItem(29);
        ___qtablewidgetitem31->setText(QCoreApplication::translate("InputRZForm", "30", nullptr));
        QTableWidgetItem *___qtablewidgetitem32 = sproutTable->verticalHeaderItem(30);
        ___qtablewidgetitem32->setText(QCoreApplication::translate("InputRZForm", "31", nullptr));
        QTableWidgetItem *___qtablewidgetitem33 = sproutTable->verticalHeaderItem(31);
        ___qtablewidgetitem33->setText(QCoreApplication::translate("InputRZForm", "32", nullptr));
        QTableWidgetItem *___qtablewidgetitem34 = sproutTable->verticalHeaderItem(32);
        ___qtablewidgetitem34->setText(QCoreApplication::translate("InputRZForm", "33", nullptr));
        QTableWidgetItem *___qtablewidgetitem35 = sproutTable->verticalHeaderItem(33);
        ___qtablewidgetitem35->setText(QCoreApplication::translate("InputRZForm", "34", nullptr));
        QTableWidgetItem *___qtablewidgetitem36 = sproutTable->verticalHeaderItem(34);
        ___qtablewidgetitem36->setText(QCoreApplication::translate("InputRZForm", "35", nullptr));
        QTableWidgetItem *___qtablewidgetitem37 = sproutTable->verticalHeaderItem(35);
        ___qtablewidgetitem37->setText(QCoreApplication::translate("InputRZForm", "36", nullptr));
        QTableWidgetItem *___qtablewidgetitem38 = sproutTable->verticalHeaderItem(36);
        ___qtablewidgetitem38->setText(QCoreApplication::translate("InputRZForm", "37", nullptr));
        QTableWidgetItem *___qtablewidgetitem39 = sproutTable->verticalHeaderItem(37);
        ___qtablewidgetitem39->setText(QCoreApplication::translate("InputRZForm", "38", nullptr));
        QTableWidgetItem *___qtablewidgetitem40 = sproutTable->verticalHeaderItem(38);
        ___qtablewidgetitem40->setText(QCoreApplication::translate("InputRZForm", "39", nullptr));
        QTableWidgetItem *___qtablewidgetitem41 = sproutTable->verticalHeaderItem(39);
        ___qtablewidgetitem41->setText(QCoreApplication::translate("InputRZForm", "40", nullptr));
        QTableWidgetItem *___qtablewidgetitem42 = sproutTable->verticalHeaderItem(40);
        ___qtablewidgetitem42->setText(QCoreApplication::translate("InputRZForm", "41", nullptr));
        QTableWidgetItem *___qtablewidgetitem43 = sproutTable->verticalHeaderItem(41);
        ___qtablewidgetitem43->setText(QCoreApplication::translate("InputRZForm", "42", nullptr));
        QTableWidgetItem *___qtablewidgetitem44 = sproutTable->verticalHeaderItem(42);
        ___qtablewidgetitem44->setText(QCoreApplication::translate("InputRZForm", "43", nullptr));
        QTableWidgetItem *___qtablewidgetitem45 = sproutTable->verticalHeaderItem(43);
        ___qtablewidgetitem45->setText(QCoreApplication::translate("InputRZForm", "44", nullptr));
        QTableWidgetItem *___qtablewidgetitem46 = sproutTable->verticalHeaderItem(44);
        ___qtablewidgetitem46->setText(QCoreApplication::translate("InputRZForm", "45", nullptr));
        QTableWidgetItem *___qtablewidgetitem47 = sproutTable->verticalHeaderItem(45);
        ___qtablewidgetitem47->setText(QCoreApplication::translate("InputRZForm", "46", nullptr));
        QTableWidgetItem *___qtablewidgetitem48 = sproutTable->verticalHeaderItem(46);
        ___qtablewidgetitem48->setText(QCoreApplication::translate("InputRZForm", "47", nullptr));
        QTableWidgetItem *___qtablewidgetitem49 = sproutTable->verticalHeaderItem(47);
        ___qtablewidgetitem49->setText(QCoreApplication::translate("InputRZForm", "48", nullptr));
        QTableWidgetItem *___qtablewidgetitem50 = sproutTable->verticalHeaderItem(48);
        ___qtablewidgetitem50->setText(QCoreApplication::translate("InputRZForm", "49", nullptr));
        QTableWidgetItem *___qtablewidgetitem51 = sproutTable->verticalHeaderItem(49);
        ___qtablewidgetitem51->setText(QCoreApplication::translate("InputRZForm", "50", nullptr));
#if QT_CONFIG(tooltip)
        sproutTable->setToolTip(QCoreApplication::translate("InputRZForm", "Enter regions in which stopping power ratios are to be output.", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        sproutTable->setWhatsThis(QCoreApplication::translate("InputRZForm", "Enter regions in which stopping power ratios are to be output.", nullptr));
#endif // QT_CONFIG(whatsthis)
        PrintFluSpecGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Print fluence spectra", nullptr));
        PrintFluSpeComboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "all", nullptr));
        PrintFluSpeComboBox->setItemText(1, QCoreApplication::translate("InputRZForm", "specified", nullptr));
        PrintFluSpeComboBox->setItemText(2, QCoreApplication::translate("InputRZForm", "none", nullptr));
        PrintFluSpeComboBox->setItemText(3, QCoreApplication::translate("InputRZForm", "zut alors", nullptr));

#if QT_CONFIG(tooltip)
        PrintFluSpeComboBox->setToolTip(QCoreApplication::translate("InputRZForm", "Define wheter spectra will be printed for all regions, specific regions or none.", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        PrintFluSpeComboBox->setWhatsThis(QCoreApplication::translate("InputRZForm", "Define wheter spectra will be printed for all regions, specific regions or none.", nullptr));
#endif // QT_CONFIG(whatsthis)
        QTableWidgetItem *___qtablewidgetitem52 = ListFluTable->horizontalHeaderItem(0);
        ___qtablewidgetitem52->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem53 = ListFluTable->horizontalHeaderItem(1);
        ___qtablewidgetitem53->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem54 = ListFluTable->verticalHeaderItem(0);
        ___qtablewidgetitem54->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem55 = ListFluTable->verticalHeaderItem(1);
        ___qtablewidgetitem55->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem56 = ListFluTable->verticalHeaderItem(2);
        ___qtablewidgetitem56->setText(QCoreApplication::translate("InputRZForm", "3", nullptr));
        QTableWidgetItem *___qtablewidgetitem57 = ListFluTable->verticalHeaderItem(3);
        ___qtablewidgetitem57->setText(QCoreApplication::translate("InputRZForm", "4", nullptr));
        QTableWidgetItem *___qtablewidgetitem58 = ListFluTable->verticalHeaderItem(4);
        ___qtablewidgetitem58->setText(QCoreApplication::translate("InputRZForm", "5", nullptr));
        QTableWidgetItem *___qtablewidgetitem59 = ListFluTable->verticalHeaderItem(5);
        ___qtablewidgetitem59->setText(QCoreApplication::translate("InputRZForm", "6", nullptr));
        QTableWidgetItem *___qtablewidgetitem60 = ListFluTable->verticalHeaderItem(6);
        ___qtablewidgetitem60->setText(QCoreApplication::translate("InputRZForm", "7", nullptr));
        QTableWidgetItem *___qtablewidgetitem61 = ListFluTable->verticalHeaderItem(7);
        ___qtablewidgetitem61->setText(QCoreApplication::translate("InputRZForm", "8", nullptr));
        QTableWidgetItem *___qtablewidgetitem62 = ListFluTable->verticalHeaderItem(8);
        ___qtablewidgetitem62->setText(QCoreApplication::translate("InputRZForm", "9", nullptr));
        QTableWidgetItem *___qtablewidgetitem63 = ListFluTable->verticalHeaderItem(9);
        ___qtablewidgetitem63->setText(QCoreApplication::translate("InputRZForm", "10", nullptr));
        QTableWidgetItem *___qtablewidgetitem64 = ListFluTable->verticalHeaderItem(10);
        ___qtablewidgetitem64->setText(QCoreApplication::translate("InputRZForm", "11", nullptr));
        QTableWidgetItem *___qtablewidgetitem65 = ListFluTable->verticalHeaderItem(11);
        ___qtablewidgetitem65->setText(QCoreApplication::translate("InputRZForm", "12", nullptr));
        QTableWidgetItem *___qtablewidgetitem66 = ListFluTable->verticalHeaderItem(12);
        ___qtablewidgetitem66->setText(QCoreApplication::translate("InputRZForm", "13", nullptr));
        QTableWidgetItem *___qtablewidgetitem67 = ListFluTable->verticalHeaderItem(13);
        ___qtablewidgetitem67->setText(QCoreApplication::translate("InputRZForm", "14", nullptr));
        QTableWidgetItem *___qtablewidgetitem68 = ListFluTable->verticalHeaderItem(14);
        ___qtablewidgetitem68->setText(QCoreApplication::translate("InputRZForm", "15", nullptr));
        QTableWidgetItem *___qtablewidgetitem69 = ListFluTable->verticalHeaderItem(15);
        ___qtablewidgetitem69->setText(QCoreApplication::translate("InputRZForm", "16", nullptr));
        QTableWidgetItem *___qtablewidgetitem70 = ListFluTable->verticalHeaderItem(16);
        ___qtablewidgetitem70->setText(QCoreApplication::translate("InputRZForm", "17", nullptr));
        QTableWidgetItem *___qtablewidgetitem71 = ListFluTable->verticalHeaderItem(17);
        ___qtablewidgetitem71->setText(QCoreApplication::translate("InputRZForm", "18", nullptr));
        QTableWidgetItem *___qtablewidgetitem72 = ListFluTable->verticalHeaderItem(18);
        ___qtablewidgetitem72->setText(QCoreApplication::translate("InputRZForm", "19", nullptr));
        QTableWidgetItem *___qtablewidgetitem73 = ListFluTable->verticalHeaderItem(19);
        ___qtablewidgetitem73->setText(QCoreApplication::translate("InputRZForm", "20", nullptr));
        QTableWidgetItem *___qtablewidgetitem74 = ListFluTable->verticalHeaderItem(20);
        ___qtablewidgetitem74->setText(QCoreApplication::translate("InputRZForm", "21", nullptr));
        QTableWidgetItem *___qtablewidgetitem75 = ListFluTable->verticalHeaderItem(21);
        ___qtablewidgetitem75->setText(QCoreApplication::translate("InputRZForm", "22", nullptr));
        QTableWidgetItem *___qtablewidgetitem76 = ListFluTable->verticalHeaderItem(22);
        ___qtablewidgetitem76->setText(QCoreApplication::translate("InputRZForm", "23", nullptr));
        QTableWidgetItem *___qtablewidgetitem77 = ListFluTable->verticalHeaderItem(23);
        ___qtablewidgetitem77->setText(QCoreApplication::translate("InputRZForm", "24", nullptr));
        QTableWidgetItem *___qtablewidgetitem78 = ListFluTable->verticalHeaderItem(24);
        ___qtablewidgetitem78->setText(QCoreApplication::translate("InputRZForm", "25", nullptr));
        QTableWidgetItem *___qtablewidgetitem79 = ListFluTable->verticalHeaderItem(25);
        ___qtablewidgetitem79->setText(QCoreApplication::translate("InputRZForm", "26", nullptr));
        QTableWidgetItem *___qtablewidgetitem80 = ListFluTable->verticalHeaderItem(26);
        ___qtablewidgetitem80->setText(QCoreApplication::translate("InputRZForm", "27", nullptr));
        QTableWidgetItem *___qtablewidgetitem81 = ListFluTable->verticalHeaderItem(27);
        ___qtablewidgetitem81->setText(QCoreApplication::translate("InputRZForm", "28", nullptr));
        QTableWidgetItem *___qtablewidgetitem82 = ListFluTable->verticalHeaderItem(28);
        ___qtablewidgetitem82->setText(QCoreApplication::translate("InputRZForm", "29", nullptr));
        QTableWidgetItem *___qtablewidgetitem83 = ListFluTable->verticalHeaderItem(29);
        ___qtablewidgetitem83->setText(QCoreApplication::translate("InputRZForm", "30", nullptr));
        QTableWidgetItem *___qtablewidgetitem84 = ListFluTable->verticalHeaderItem(30);
        ___qtablewidgetitem84->setText(QCoreApplication::translate("InputRZForm", "31", nullptr));
        QTableWidgetItem *___qtablewidgetitem85 = ListFluTable->verticalHeaderItem(31);
        ___qtablewidgetitem85->setText(QCoreApplication::translate("InputRZForm", "32", nullptr));
        QTableWidgetItem *___qtablewidgetitem86 = ListFluTable->verticalHeaderItem(32);
        ___qtablewidgetitem86->setText(QCoreApplication::translate("InputRZForm", "33", nullptr));
        QTableWidgetItem *___qtablewidgetitem87 = ListFluTable->verticalHeaderItem(33);
        ___qtablewidgetitem87->setText(QCoreApplication::translate("InputRZForm", "34", nullptr));
        QTableWidgetItem *___qtablewidgetitem88 = ListFluTable->verticalHeaderItem(34);
        ___qtablewidgetitem88->setText(QCoreApplication::translate("InputRZForm", "35", nullptr));
        QTableWidgetItem *___qtablewidgetitem89 = ListFluTable->verticalHeaderItem(35);
        ___qtablewidgetitem89->setText(QCoreApplication::translate("InputRZForm", "36", nullptr));
        QTableWidgetItem *___qtablewidgetitem90 = ListFluTable->verticalHeaderItem(36);
        ___qtablewidgetitem90->setText(QCoreApplication::translate("InputRZForm", "37", nullptr));
        QTableWidgetItem *___qtablewidgetitem91 = ListFluTable->verticalHeaderItem(37);
        ___qtablewidgetitem91->setText(QCoreApplication::translate("InputRZForm", "38", nullptr));
        QTableWidgetItem *___qtablewidgetitem92 = ListFluTable->verticalHeaderItem(38);
        ___qtablewidgetitem92->setText(QCoreApplication::translate("InputRZForm", "39", nullptr));
        QTableWidgetItem *___qtablewidgetitem93 = ListFluTable->verticalHeaderItem(39);
        ___qtablewidgetitem93->setText(QCoreApplication::translate("InputRZForm", "40", nullptr));
        QTableWidgetItem *___qtablewidgetitem94 = ListFluTable->verticalHeaderItem(40);
        ___qtablewidgetitem94->setText(QCoreApplication::translate("InputRZForm", "41", nullptr));
        QTableWidgetItem *___qtablewidgetitem95 = ListFluTable->verticalHeaderItem(41);
        ___qtablewidgetitem95->setText(QCoreApplication::translate("InputRZForm", "42", nullptr));
        QTableWidgetItem *___qtablewidgetitem96 = ListFluTable->verticalHeaderItem(42);
        ___qtablewidgetitem96->setText(QCoreApplication::translate("InputRZForm", "43", nullptr));
        QTableWidgetItem *___qtablewidgetitem97 = ListFluTable->verticalHeaderItem(43);
        ___qtablewidgetitem97->setText(QCoreApplication::translate("InputRZForm", "44", nullptr));
        QTableWidgetItem *___qtablewidgetitem98 = ListFluTable->verticalHeaderItem(44);
        ___qtablewidgetitem98->setText(QCoreApplication::translate("InputRZForm", "45", nullptr));
        QTableWidgetItem *___qtablewidgetitem99 = ListFluTable->verticalHeaderItem(45);
        ___qtablewidgetitem99->setText(QCoreApplication::translate("InputRZForm", "46", nullptr));
        QTableWidgetItem *___qtablewidgetitem100 = ListFluTable->verticalHeaderItem(46);
        ___qtablewidgetitem100->setText(QCoreApplication::translate("InputRZForm", "47", nullptr));
        QTableWidgetItem *___qtablewidgetitem101 = ListFluTable->verticalHeaderItem(47);
        ___qtablewidgetitem101->setText(QCoreApplication::translate("InputRZForm", "48", nullptr));
        QTableWidgetItem *___qtablewidgetitem102 = ListFluTable->verticalHeaderItem(48);
        ___qtablewidgetitem102->setText(QCoreApplication::translate("InputRZForm", "49", nullptr));
        QTableWidgetItem *___qtablewidgetitem103 = ListFluTable->verticalHeaderItem(49);
        ___qtablewidgetitem103->setText(QCoreApplication::translate("InputRZForm", "50", nullptr));
#if QT_CONFIG(tooltip)
        ListFluTable->setToolTip(QCoreApplication::translate("InputRZForm", "Specify regions to print spectra.", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        ListFluTable->setWhatsThis(QCoreApplication::translate("InputRZForm", "Specify regions to print spectra.", nullptr));
#endif // QT_CONFIG(whatsthis)
#if QT_CONFIG(tooltip)
        IPRIMARYGroupBox->setToolTip(QCoreApplication::translate("InputRZForm", "Defines for which particles spectra are going to be printed.", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        IPRIMARYGroupBox->setWhatsThis(QCoreApplication::translate("InputRZForm", "Defines for which particles spectra are going to be printed.", nullptr));
#endif // QT_CONFIG(whatsthis)
        IPRIMARYGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Type of spectra to print", nullptr));
        IPRIMARYComboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "total fluence", nullptr));
        IPRIMARYComboBox->setItemText(1, QCoreApplication::translate("InputRZForm", "electron primaries", nullptr));
        IPRIMARYComboBox->setItemText(2, QCoreApplication::translate("InputRZForm", "include brem secondaries", nullptr));
        IPRIMARYComboBox->setItemText(3, QCoreApplication::translate("InputRZForm", "photon primaries", nullptr));
        IPRIMARYComboBox->setItemText(4, QCoreApplication::translate("InputRZForm", "electron secondaries", nullptr));

#if QT_CONFIG(tooltip)
        IPRIMARYComboBox->setToolTip(QCoreApplication::translate("InputRZForm", "Defines for which particles spectra are going to be printed.", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        IPRIMARYComboBox->setWhatsThis(QCoreApplication::translate("InputRZForm", "Defines for which particles spectra are going to be printed.", nullptr));
#endif // QT_CONFIG(whatsthis)
        sloteFluGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Output energy bins", nullptr));
        sloteFluLabel->setText(QCoreApplication::translate("InputRZForm", "SLOTE", nullptr));
        QTableWidgetItem *___qtablewidgetitem104 = sloteFluTable->horizontalHeaderItem(0);
        ___qtablewidgetitem104->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem105 = sloteFluTable->verticalHeaderItem(0);
        ___qtablewidgetitem105->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem106 = sloteFluTable->verticalHeaderItem(1);
        ___qtablewidgetitem106->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem107 = sloteFluTable->verticalHeaderItem(2);
        ___qtablewidgetitem107->setText(QCoreApplication::translate("InputRZForm", "3", nullptr));
        QTableWidgetItem *___qtablewidgetitem108 = sloteFluTable->verticalHeaderItem(3);
        ___qtablewidgetitem108->setText(QCoreApplication::translate("InputRZForm", "4", nullptr));
        QTableWidgetItem *___qtablewidgetitem109 = sloteFluTable->verticalHeaderItem(4);
        ___qtablewidgetitem109->setText(QCoreApplication::translate("InputRZForm", "5", nullptr));
        QTableWidgetItem *___qtablewidgetitem110 = sloteFluTable->verticalHeaderItem(5);
        ___qtablewidgetitem110->setText(QCoreApplication::translate("InputRZForm", "6", nullptr));
        QTableWidgetItem *___qtablewidgetitem111 = sloteFluTable->verticalHeaderItem(6);
        ___qtablewidgetitem111->setText(QCoreApplication::translate("InputRZForm", "7", nullptr));
        QTableWidgetItem *___qtablewidgetitem112 = sloteFluTable->verticalHeaderItem(7);
        ___qtablewidgetitem112->setText(QCoreApplication::translate("InputRZForm", "8", nullptr));
        QTableWidgetItem *___qtablewidgetitem113 = sloteFluTable->verticalHeaderItem(8);
        ___qtablewidgetitem113->setText(QCoreApplication::translate("InputRZForm", "9", nullptr));
        QTableWidgetItem *___qtablewidgetitem114 = sloteFluTable->verticalHeaderItem(9);
        ___qtablewidgetitem114->setText(QCoreApplication::translate("InputRZForm", "10", nullptr));
        QTableWidgetItem *___qtablewidgetitem115 = sloteFluTable->verticalHeaderItem(10);
        ___qtablewidgetitem115->setText(QCoreApplication::translate("InputRZForm", "11", nullptr));
        QTableWidgetItem *___qtablewidgetitem116 = sloteFluTable->verticalHeaderItem(11);
        ___qtablewidgetitem116->setText(QCoreApplication::translate("InputRZForm", "12", nullptr));
        QTableWidgetItem *___qtablewidgetitem117 = sloteFluTable->verticalHeaderItem(12);
        ___qtablewidgetitem117->setText(QCoreApplication::translate("InputRZForm", "13", nullptr));
        QTableWidgetItem *___qtablewidgetitem118 = sloteFluTable->verticalHeaderItem(13);
        ___qtablewidgetitem118->setText(QCoreApplication::translate("InputRZForm", "14", nullptr));
        QTableWidgetItem *___qtablewidgetitem119 = sloteFluTable->verticalHeaderItem(14);
        ___qtablewidgetitem119->setText(QCoreApplication::translate("InputRZForm", "15", nullptr));
        QTableWidgetItem *___qtablewidgetitem120 = sloteFluTable->verticalHeaderItem(15);
        ___qtablewidgetitem120->setText(QCoreApplication::translate("InputRZForm", "16", nullptr));
        QTableWidgetItem *___qtablewidgetitem121 = sloteFluTable->verticalHeaderItem(16);
        ___qtablewidgetitem121->setText(QCoreApplication::translate("InputRZForm", "17", nullptr));
        QTableWidgetItem *___qtablewidgetitem122 = sloteFluTable->verticalHeaderItem(17);
        ___qtablewidgetitem122->setText(QCoreApplication::translate("InputRZForm", "18", nullptr));
        QTableWidgetItem *___qtablewidgetitem123 = sloteFluTable->verticalHeaderItem(18);
        ___qtablewidgetitem123->setText(QCoreApplication::translate("InputRZForm", "19", nullptr));
        QTableWidgetItem *___qtablewidgetitem124 = sloteFluTable->verticalHeaderItem(19);
        ___qtablewidgetitem124->setText(QCoreApplication::translate("InputRZForm", "20", nullptr));
        QTableWidgetItem *___qtablewidgetitem125 = sloteFluTable->verticalHeaderItem(20);
        ___qtablewidgetitem125->setText(QCoreApplication::translate("InputRZForm", "21", nullptr));
        QTableWidgetItem *___qtablewidgetitem126 = sloteFluTable->verticalHeaderItem(21);
        ___qtablewidgetitem126->setText(QCoreApplication::translate("InputRZForm", "22", nullptr));
        QTableWidgetItem *___qtablewidgetitem127 = sloteFluTable->verticalHeaderItem(22);
        ___qtablewidgetitem127->setText(QCoreApplication::translate("InputRZForm", "23", nullptr));
        QTableWidgetItem *___qtablewidgetitem128 = sloteFluTable->verticalHeaderItem(23);
        ___qtablewidgetitem128->setText(QCoreApplication::translate("InputRZForm", "24", nullptr));
        QTableWidgetItem *___qtablewidgetitem129 = sloteFluTable->verticalHeaderItem(24);
        ___qtablewidgetitem129->setText(QCoreApplication::translate("InputRZForm", "25", nullptr));
        QTableWidgetItem *___qtablewidgetitem130 = sloteFluTable->verticalHeaderItem(25);
        ___qtablewidgetitem130->setText(QCoreApplication::translate("InputRZForm", "26", nullptr));
        QTableWidgetItem *___qtablewidgetitem131 = sloteFluTable->verticalHeaderItem(26);
        ___qtablewidgetitem131->setText(QCoreApplication::translate("InputRZForm", "27", nullptr));
        QTableWidgetItem *___qtablewidgetitem132 = sloteFluTable->verticalHeaderItem(27);
        ___qtablewidgetitem132->setText(QCoreApplication::translate("InputRZForm", "28", nullptr));
        QTableWidgetItem *___qtablewidgetitem133 = sloteFluTable->verticalHeaderItem(28);
        ___qtablewidgetitem133->setText(QCoreApplication::translate("InputRZForm", "29", nullptr));
        QTableWidgetItem *___qtablewidgetitem134 = sloteFluTable->verticalHeaderItem(29);
        ___qtablewidgetitem134->setText(QCoreApplication::translate("InputRZForm", "30", nullptr));
        QTableWidgetItem *___qtablewidgetitem135 = sloteFluTable->verticalHeaderItem(30);
        ___qtablewidgetitem135->setText(QCoreApplication::translate("InputRZForm", "31", nullptr));
        QTableWidgetItem *___qtablewidgetitem136 = sloteFluTable->verticalHeaderItem(31);
        ___qtablewidgetitem136->setText(QCoreApplication::translate("InputRZForm", "32", nullptr));
        QTableWidgetItem *___qtablewidgetitem137 = sloteFluTable->verticalHeaderItem(32);
        ___qtablewidgetitem137->setText(QCoreApplication::translate("InputRZForm", "33", nullptr));
        QTableWidgetItem *___qtablewidgetitem138 = sloteFluTable->verticalHeaderItem(33);
        ___qtablewidgetitem138->setText(QCoreApplication::translate("InputRZForm", "34", nullptr));
        QTableWidgetItem *___qtablewidgetitem139 = sloteFluTable->verticalHeaderItem(34);
        ___qtablewidgetitem139->setText(QCoreApplication::translate("InputRZForm", "35", nullptr));
        QTableWidgetItem *___qtablewidgetitem140 = sloteFluTable->verticalHeaderItem(35);
        ___qtablewidgetitem140->setText(QCoreApplication::translate("InputRZForm", "36", nullptr));
        QTableWidgetItem *___qtablewidgetitem141 = sloteFluTable->verticalHeaderItem(36);
        ___qtablewidgetitem141->setText(QCoreApplication::translate("InputRZForm", "37", nullptr));
        QTableWidgetItem *___qtablewidgetitem142 = sloteFluTable->verticalHeaderItem(37);
        ___qtablewidgetitem142->setText(QCoreApplication::translate("InputRZForm", "38", nullptr));
        QTableWidgetItem *___qtablewidgetitem143 = sloteFluTable->verticalHeaderItem(38);
        ___qtablewidgetitem143->setText(QCoreApplication::translate("InputRZForm", "39", nullptr));
        QTableWidgetItem *___qtablewidgetitem144 = sloteFluTable->verticalHeaderItem(39);
        ___qtablewidgetitem144->setText(QCoreApplication::translate("InputRZForm", "40", nullptr));
        QTableWidgetItem *___qtablewidgetitem145 = sloteFluTable->verticalHeaderItem(40);
        ___qtablewidgetitem145->setText(QCoreApplication::translate("InputRZForm", "41", nullptr));
        QTableWidgetItem *___qtablewidgetitem146 = sloteFluTable->verticalHeaderItem(41);
        ___qtablewidgetitem146->setText(QCoreApplication::translate("InputRZForm", "42", nullptr));
        QTableWidgetItem *___qtablewidgetitem147 = sloteFluTable->verticalHeaderItem(42);
        ___qtablewidgetitem147->setText(QCoreApplication::translate("InputRZForm", "43", nullptr));
        QTableWidgetItem *___qtablewidgetitem148 = sloteFluTable->verticalHeaderItem(43);
        ___qtablewidgetitem148->setText(QCoreApplication::translate("InputRZForm", "44", nullptr));
        QTableWidgetItem *___qtablewidgetitem149 = sloteFluTable->verticalHeaderItem(44);
        ___qtablewidgetitem149->setText(QCoreApplication::translate("InputRZForm", "45", nullptr));
        QTableWidgetItem *___qtablewidgetitem150 = sloteFluTable->verticalHeaderItem(45);
        ___qtablewidgetitem150->setText(QCoreApplication::translate("InputRZForm", "46", nullptr));
        QTableWidgetItem *___qtablewidgetitem151 = sloteFluTable->verticalHeaderItem(46);
        ___qtablewidgetitem151->setText(QCoreApplication::translate("InputRZForm", "47", nullptr));
        QTableWidgetItem *___qtablewidgetitem152 = sloteFluTable->verticalHeaderItem(47);
        ___qtablewidgetitem152->setText(QCoreApplication::translate("InputRZForm", "48", nullptr));
        QTableWidgetItem *___qtablewidgetitem153 = sloteFluTable->verticalHeaderItem(48);
        ___qtablewidgetitem153->setText(QCoreApplication::translate("InputRZForm", "49", nullptr));
        QTableWidgetItem *___qtablewidgetitem154 = sloteFluTable->verticalHeaderItem(49);
        ___qtablewidgetitem154->setText(QCoreApplication::translate("InputRZForm", "50", nullptr));
        QTableWidgetItem *___qtablewidgetitem155 = sloteFluTable->verticalHeaderItem(50);
        ___qtablewidgetitem155->setText(QCoreApplication::translate("InputRZForm", "51", nullptr));
        QTableWidgetItem *___qtablewidgetitem156 = sloteFluTable->verticalHeaderItem(51);
        ___qtablewidgetitem156->setText(QCoreApplication::translate("InputRZForm", "52", nullptr));
        QTableWidgetItem *___qtablewidgetitem157 = sloteFluTable->verticalHeaderItem(52);
        ___qtablewidgetitem157->setText(QCoreApplication::translate("InputRZForm", "53", nullptr));
        QTableWidgetItem *___qtablewidgetitem158 = sloteFluTable->verticalHeaderItem(53);
        ___qtablewidgetitem158->setText(QCoreApplication::translate("InputRZForm", "54", nullptr));
        QTableWidgetItem *___qtablewidgetitem159 = sloteFluTable->verticalHeaderItem(54);
        ___qtablewidgetitem159->setText(QCoreApplication::translate("InputRZForm", "55", nullptr));
        QTableWidgetItem *___qtablewidgetitem160 = sloteFluTable->verticalHeaderItem(55);
        ___qtablewidgetitem160->setText(QCoreApplication::translate("InputRZForm", "56", nullptr));
        QTableWidgetItem *___qtablewidgetitem161 = sloteFluTable->verticalHeaderItem(56);
        ___qtablewidgetitem161->setText(QCoreApplication::translate("InputRZForm", "57", nullptr));
        QTableWidgetItem *___qtablewidgetitem162 = sloteFluTable->verticalHeaderItem(57);
        ___qtablewidgetitem162->setText(QCoreApplication::translate("InputRZForm", "58", nullptr));
        QTableWidgetItem *___qtablewidgetitem163 = sloteFluTable->verticalHeaderItem(58);
        ___qtablewidgetitem163->setText(QCoreApplication::translate("InputRZForm", "59", nullptr));
        QTableWidgetItem *___qtablewidgetitem164 = sloteFluTable->verticalHeaderItem(59);
        ___qtablewidgetitem164->setText(QCoreApplication::translate("InputRZForm", "60", nullptr));
        QTableWidgetItem *___qtablewidgetitem165 = sloteFluTable->verticalHeaderItem(60);
        ___qtablewidgetitem165->setText(QCoreApplication::translate("InputRZForm", "61", nullptr));
        QTableWidgetItem *___qtablewidgetitem166 = sloteFluTable->verticalHeaderItem(61);
        ___qtablewidgetitem166->setText(QCoreApplication::translate("InputRZForm", "62", nullptr));
        QTableWidgetItem *___qtablewidgetitem167 = sloteFluTable->verticalHeaderItem(62);
        ___qtablewidgetitem167->setText(QCoreApplication::translate("InputRZForm", "63", nullptr));
        QTableWidgetItem *___qtablewidgetitem168 = sloteFluTable->verticalHeaderItem(63);
        ___qtablewidgetitem168->setText(QCoreApplication::translate("InputRZForm", "64", nullptr));
        QTableWidgetItem *___qtablewidgetitem169 = sloteFluTable->verticalHeaderItem(64);
        ___qtablewidgetitem169->setText(QCoreApplication::translate("InputRZForm", "65", nullptr));
        QTableWidgetItem *___qtablewidgetitem170 = sloteFluTable->verticalHeaderItem(65);
        ___qtablewidgetitem170->setText(QCoreApplication::translate("InputRZForm", "66", nullptr));
        QTableWidgetItem *___qtablewidgetitem171 = sloteFluTable->verticalHeaderItem(66);
        ___qtablewidgetitem171->setText(QCoreApplication::translate("InputRZForm", "67", nullptr));
        QTableWidgetItem *___qtablewidgetitem172 = sloteFluTable->verticalHeaderItem(67);
        ___qtablewidgetitem172->setText(QCoreApplication::translate("InputRZForm", "68", nullptr));
        QTableWidgetItem *___qtablewidgetitem173 = sloteFluTable->verticalHeaderItem(68);
        ___qtablewidgetitem173->setText(QCoreApplication::translate("InputRZForm", "69", nullptr));
        QTableWidgetItem *___qtablewidgetitem174 = sloteFluTable->verticalHeaderItem(69);
        ___qtablewidgetitem174->setText(QCoreApplication::translate("InputRZForm", "70", nullptr));
        QTableWidgetItem *___qtablewidgetitem175 = sloteFluTable->verticalHeaderItem(70);
        ___qtablewidgetitem175->setText(QCoreApplication::translate("InputRZForm", "71", nullptr));
        QTableWidgetItem *___qtablewidgetitem176 = sloteFluTable->verticalHeaderItem(71);
        ___qtablewidgetitem176->setText(QCoreApplication::translate("InputRZForm", "72", nullptr));
        QTableWidgetItem *___qtablewidgetitem177 = sloteFluTable->verticalHeaderItem(72);
        ___qtablewidgetitem177->setText(QCoreApplication::translate("InputRZForm", "73", nullptr));
        QTableWidgetItem *___qtablewidgetitem178 = sloteFluTable->verticalHeaderItem(73);
        ___qtablewidgetitem178->setText(QCoreApplication::translate("InputRZForm", "74", nullptr));
        QTableWidgetItem *___qtablewidgetitem179 = sloteFluTable->verticalHeaderItem(74);
        ___qtablewidgetitem179->setText(QCoreApplication::translate("InputRZForm", "75", nullptr));
        QTableWidgetItem *___qtablewidgetitem180 = sloteFluTable->verticalHeaderItem(75);
        ___qtablewidgetitem180->setText(QCoreApplication::translate("InputRZForm", "76", nullptr));
        QTableWidgetItem *___qtablewidgetitem181 = sloteFluTable->verticalHeaderItem(76);
        ___qtablewidgetitem181->setText(QCoreApplication::translate("InputRZForm", "77", nullptr));
        QTableWidgetItem *___qtablewidgetitem182 = sloteFluTable->verticalHeaderItem(77);
        ___qtablewidgetitem182->setText(QCoreApplication::translate("InputRZForm", "78", nullptr));
        QTableWidgetItem *___qtablewidgetitem183 = sloteFluTable->verticalHeaderItem(78);
        ___qtablewidgetitem183->setText(QCoreApplication::translate("InputRZForm", "79", nullptr));
        QTableWidgetItem *___qtablewidgetitem184 = sloteFluTable->verticalHeaderItem(79);
        ___qtablewidgetitem184->setText(QCoreApplication::translate("InputRZForm", "80", nullptr));
        QTableWidgetItem *___qtablewidgetitem185 = sloteFluTable->verticalHeaderItem(80);
        ___qtablewidgetitem185->setText(QCoreApplication::translate("InputRZForm", "81", nullptr));
        QTableWidgetItem *___qtablewidgetitem186 = sloteFluTable->verticalHeaderItem(81);
        ___qtablewidgetitem186->setText(QCoreApplication::translate("InputRZForm", "82", nullptr));
        QTableWidgetItem *___qtablewidgetitem187 = sloteFluTable->verticalHeaderItem(82);
        ___qtablewidgetitem187->setText(QCoreApplication::translate("InputRZForm", "83", nullptr));
        QTableWidgetItem *___qtablewidgetitem188 = sloteFluTable->verticalHeaderItem(83);
        ___qtablewidgetitem188->setText(QCoreApplication::translate("InputRZForm", "84", nullptr));
        QTableWidgetItem *___qtablewidgetitem189 = sloteFluTable->verticalHeaderItem(84);
        ___qtablewidgetitem189->setText(QCoreApplication::translate("InputRZForm", "85", nullptr));
        QTableWidgetItem *___qtablewidgetitem190 = sloteFluTable->verticalHeaderItem(85);
        ___qtablewidgetitem190->setText(QCoreApplication::translate("InputRZForm", "86", nullptr));
        QTableWidgetItem *___qtablewidgetitem191 = sloteFluTable->verticalHeaderItem(86);
        ___qtablewidgetitem191->setText(QCoreApplication::translate("InputRZForm", "87", nullptr));
        QTableWidgetItem *___qtablewidgetitem192 = sloteFluTable->verticalHeaderItem(87);
        ___qtablewidgetitem192->setText(QCoreApplication::translate("InputRZForm", "88", nullptr));
        QTableWidgetItem *___qtablewidgetitem193 = sloteFluTable->verticalHeaderItem(88);
        ___qtablewidgetitem193->setText(QCoreApplication::translate("InputRZForm", "89", nullptr));
        QTableWidgetItem *___qtablewidgetitem194 = sloteFluTable->verticalHeaderItem(89);
        ___qtablewidgetitem194->setText(QCoreApplication::translate("InputRZForm", "90", nullptr));
        QTableWidgetItem *___qtablewidgetitem195 = sloteFluTable->verticalHeaderItem(90);
        ___qtablewidgetitem195->setText(QCoreApplication::translate("InputRZForm", "91", nullptr));
        QTableWidgetItem *___qtablewidgetitem196 = sloteFluTable->verticalHeaderItem(91);
        ___qtablewidgetitem196->setText(QCoreApplication::translate("InputRZForm", "92", nullptr));
        QTableWidgetItem *___qtablewidgetitem197 = sloteFluTable->verticalHeaderItem(92);
        ___qtablewidgetitem197->setText(QCoreApplication::translate("InputRZForm", "93", nullptr));
        QTableWidgetItem *___qtablewidgetitem198 = sloteFluTable->verticalHeaderItem(93);
        ___qtablewidgetitem198->setText(QCoreApplication::translate("InputRZForm", "94", nullptr));
        QTableWidgetItem *___qtablewidgetitem199 = sloteFluTable->verticalHeaderItem(94);
        ___qtablewidgetitem199->setText(QCoreApplication::translate("InputRZForm", "95", nullptr));
        QTableWidgetItem *___qtablewidgetitem200 = sloteFluTable->verticalHeaderItem(95);
        ___qtablewidgetitem200->setText(QCoreApplication::translate("InputRZForm", "96", nullptr));
        QTableWidgetItem *___qtablewidgetitem201 = sloteFluTable->verticalHeaderItem(96);
        ___qtablewidgetitem201->setText(QCoreApplication::translate("InputRZForm", "97", nullptr));
        QTableWidgetItem *___qtablewidgetitem202 = sloteFluTable->verticalHeaderItem(97);
        ___qtablewidgetitem202->setText(QCoreApplication::translate("InputRZForm", "98", nullptr));
        QTableWidgetItem *___qtablewidgetitem203 = sloteFluTable->verticalHeaderItem(98);
        ___qtablewidgetitem203->setText(QCoreApplication::translate("InputRZForm", "99", nullptr));
        QTableWidgetItem *___qtablewidgetitem204 = sloteFluTable->verticalHeaderItem(99);
        ___qtablewidgetitem204->setText(QCoreApplication::translate("InputRZForm", "100", nullptr));
        QTableWidgetItem *___qtablewidgetitem205 = sloteFluTable->verticalHeaderItem(100);
        ___qtablewidgetitem205->setText(QCoreApplication::translate("InputRZForm", "101", nullptr));
        QTableWidgetItem *___qtablewidgetitem206 = sloteFluTable->verticalHeaderItem(101);
        ___qtablewidgetitem206->setText(QCoreApplication::translate("InputRZForm", "102", nullptr));
        QTableWidgetItem *___qtablewidgetitem207 = sloteFluTable->verticalHeaderItem(102);
        ___qtablewidgetitem207->setText(QCoreApplication::translate("InputRZForm", "103", nullptr));
        QTableWidgetItem *___qtablewidgetitem208 = sloteFluTable->verticalHeaderItem(103);
        ___qtablewidgetitem208->setText(QCoreApplication::translate("InputRZForm", "104", nullptr));
        QTableWidgetItem *___qtablewidgetitem209 = sloteFluTable->verticalHeaderItem(104);
        ___qtablewidgetitem209->setText(QCoreApplication::translate("InputRZForm", "105", nullptr));
        QTableWidgetItem *___qtablewidgetitem210 = sloteFluTable->verticalHeaderItem(105);
        ___qtablewidgetitem210->setText(QCoreApplication::translate("InputRZForm", "106", nullptr));
        QTableWidgetItem *___qtablewidgetitem211 = sloteFluTable->verticalHeaderItem(106);
        ___qtablewidgetitem211->setText(QCoreApplication::translate("InputRZForm", "107", nullptr));
        QTableWidgetItem *___qtablewidgetitem212 = sloteFluTable->verticalHeaderItem(107);
        ___qtablewidgetitem212->setText(QCoreApplication::translate("InputRZForm", "108", nullptr));
        QTableWidgetItem *___qtablewidgetitem213 = sloteFluTable->verticalHeaderItem(108);
        ___qtablewidgetitem213->setText(QCoreApplication::translate("InputRZForm", "109", nullptr));
        QTableWidgetItem *___qtablewidgetitem214 = sloteFluTable->verticalHeaderItem(109);
        ___qtablewidgetitem214->setText(QCoreApplication::translate("InputRZForm", "110", nullptr));
        QTableWidgetItem *___qtablewidgetitem215 = sloteFluTable->verticalHeaderItem(110);
        ___qtablewidgetitem215->setText(QCoreApplication::translate("InputRZForm", "111", nullptr));
        QTableWidgetItem *___qtablewidgetitem216 = sloteFluTable->verticalHeaderItem(111);
        ___qtablewidgetitem216->setText(QCoreApplication::translate("InputRZForm", "112", nullptr));
        QTableWidgetItem *___qtablewidgetitem217 = sloteFluTable->verticalHeaderItem(112);
        ___qtablewidgetitem217->setText(QCoreApplication::translate("InputRZForm", "113", nullptr));
        QTableWidgetItem *___qtablewidgetitem218 = sloteFluTable->verticalHeaderItem(113);
        ___qtablewidgetitem218->setText(QCoreApplication::translate("InputRZForm", "114", nullptr));
        QTableWidgetItem *___qtablewidgetitem219 = sloteFluTable->verticalHeaderItem(114);
        ___qtablewidgetitem219->setText(QCoreApplication::translate("InputRZForm", "115", nullptr));
        QTableWidgetItem *___qtablewidgetitem220 = sloteFluTable->verticalHeaderItem(115);
        ___qtablewidgetitem220->setText(QCoreApplication::translate("InputRZForm", "116", nullptr));
        QTableWidgetItem *___qtablewidgetitem221 = sloteFluTable->verticalHeaderItem(116);
        ___qtablewidgetitem221->setText(QCoreApplication::translate("InputRZForm", "117", nullptr));
        QTableWidgetItem *___qtablewidgetitem222 = sloteFluTable->verticalHeaderItem(117);
        ___qtablewidgetitem222->setText(QCoreApplication::translate("InputRZForm", "118", nullptr));
        QTableWidgetItem *___qtablewidgetitem223 = sloteFluTable->verticalHeaderItem(118);
        ___qtablewidgetitem223->setText(QCoreApplication::translate("InputRZForm", "119", nullptr));
        QTableWidgetItem *___qtablewidgetitem224 = sloteFluTable->verticalHeaderItem(119);
        ___qtablewidgetitem224->setText(QCoreApplication::translate("InputRZForm", "120", nullptr));
        QTableWidgetItem *___qtablewidgetitem225 = sloteFluTable->verticalHeaderItem(120);
        ___qtablewidgetitem225->setText(QCoreApplication::translate("InputRZForm", "121", nullptr));
        QTableWidgetItem *___qtablewidgetitem226 = sloteFluTable->verticalHeaderItem(121);
        ___qtablewidgetitem226->setText(QCoreApplication::translate("InputRZForm", "122", nullptr));
        QTableWidgetItem *___qtablewidgetitem227 = sloteFluTable->verticalHeaderItem(122);
        ___qtablewidgetitem227->setText(QCoreApplication::translate("InputRZForm", "123", nullptr));
        QTableWidgetItem *___qtablewidgetitem228 = sloteFluTable->verticalHeaderItem(123);
        ___qtablewidgetitem228->setText(QCoreApplication::translate("InputRZForm", "124", nullptr));
        QTableWidgetItem *___qtablewidgetitem229 = sloteFluTable->verticalHeaderItem(124);
        ___qtablewidgetitem229->setText(QCoreApplication::translate("InputRZForm", "125", nullptr));
        QTableWidgetItem *___qtablewidgetitem230 = sloteFluTable->verticalHeaderItem(125);
        ___qtablewidgetitem230->setText(QCoreApplication::translate("InputRZForm", "126", nullptr));
        QTableWidgetItem *___qtablewidgetitem231 = sloteFluTable->verticalHeaderItem(126);
        ___qtablewidgetitem231->setText(QCoreApplication::translate("InputRZForm", "127", nullptr));
        QTableWidgetItem *___qtablewidgetitem232 = sloteFluTable->verticalHeaderItem(127);
        ___qtablewidgetitem232->setText(QCoreApplication::translate("InputRZForm", "128", nullptr));
        QTableWidgetItem *___qtablewidgetitem233 = sloteFluTable->verticalHeaderItem(128);
        ___qtablewidgetitem233->setText(QCoreApplication::translate("InputRZForm", "129", nullptr));
        QTableWidgetItem *___qtablewidgetitem234 = sloteFluTable->verticalHeaderItem(129);
        ___qtablewidgetitem234->setText(QCoreApplication::translate("InputRZForm", "130", nullptr));
        QTableWidgetItem *___qtablewidgetitem235 = sloteFluTable->verticalHeaderItem(130);
        ___qtablewidgetitem235->setText(QCoreApplication::translate("InputRZForm", "131", nullptr));
        QTableWidgetItem *___qtablewidgetitem236 = sloteFluTable->verticalHeaderItem(131);
        ___qtablewidgetitem236->setText(QCoreApplication::translate("InputRZForm", "132", nullptr));
        QTableWidgetItem *___qtablewidgetitem237 = sloteFluTable->verticalHeaderItem(132);
        ___qtablewidgetitem237->setText(QCoreApplication::translate("InputRZForm", "133", nullptr));
        QTableWidgetItem *___qtablewidgetitem238 = sloteFluTable->verticalHeaderItem(133);
        ___qtablewidgetitem238->setText(QCoreApplication::translate("InputRZForm", "134", nullptr));
        QTableWidgetItem *___qtablewidgetitem239 = sloteFluTable->verticalHeaderItem(134);
        ___qtablewidgetitem239->setText(QCoreApplication::translate("InputRZForm", "135", nullptr));
        QTableWidgetItem *___qtablewidgetitem240 = sloteFluTable->verticalHeaderItem(135);
        ___qtablewidgetitem240->setText(QCoreApplication::translate("InputRZForm", "136", nullptr));
        QTableWidgetItem *___qtablewidgetitem241 = sloteFluTable->verticalHeaderItem(136);
        ___qtablewidgetitem241->setText(QCoreApplication::translate("InputRZForm", "137", nullptr));
        QTableWidgetItem *___qtablewidgetitem242 = sloteFluTable->verticalHeaderItem(137);
        ___qtablewidgetitem242->setText(QCoreApplication::translate("InputRZForm", "138", nullptr));
        QTableWidgetItem *___qtablewidgetitem243 = sloteFluTable->verticalHeaderItem(138);
        ___qtablewidgetitem243->setText(QCoreApplication::translate("InputRZForm", "139", nullptr));
        QTableWidgetItem *___qtablewidgetitem244 = sloteFluTable->verticalHeaderItem(139);
        ___qtablewidgetitem244->setText(QCoreApplication::translate("InputRZForm", "140", nullptr));
        QTableWidgetItem *___qtablewidgetitem245 = sloteFluTable->verticalHeaderItem(140);
        ___qtablewidgetitem245->setText(QCoreApplication::translate("InputRZForm", "141", nullptr));
        QTableWidgetItem *___qtablewidgetitem246 = sloteFluTable->verticalHeaderItem(141);
        ___qtablewidgetitem246->setText(QCoreApplication::translate("InputRZForm", "142", nullptr));
        QTableWidgetItem *___qtablewidgetitem247 = sloteFluTable->verticalHeaderItem(142);
        ___qtablewidgetitem247->setText(QCoreApplication::translate("InputRZForm", "143", nullptr));
        QTableWidgetItem *___qtablewidgetitem248 = sloteFluTable->verticalHeaderItem(143);
        ___qtablewidgetitem248->setText(QCoreApplication::translate("InputRZForm", "144", nullptr));
        QTableWidgetItem *___qtablewidgetitem249 = sloteFluTable->verticalHeaderItem(144);
        ___qtablewidgetitem249->setText(QCoreApplication::translate("InputRZForm", "145", nullptr));
        QTableWidgetItem *___qtablewidgetitem250 = sloteFluTable->verticalHeaderItem(145);
        ___qtablewidgetitem250->setText(QCoreApplication::translate("InputRZForm", "146", nullptr));
        QTableWidgetItem *___qtablewidgetitem251 = sloteFluTable->verticalHeaderItem(146);
        ___qtablewidgetitem251->setText(QCoreApplication::translate("InputRZForm", "147", nullptr));
        QTableWidgetItem *___qtablewidgetitem252 = sloteFluTable->verticalHeaderItem(147);
        ___qtablewidgetitem252->setText(QCoreApplication::translate("InputRZForm", "148", nullptr));
        QTableWidgetItem *___qtablewidgetitem253 = sloteFluTable->verticalHeaderItem(148);
        ___qtablewidgetitem253->setText(QCoreApplication::translate("InputRZForm", "149", nullptr));
        QTableWidgetItem *___qtablewidgetitem254 = sloteFluTable->verticalHeaderItem(149);
        ___qtablewidgetitem254->setText(QCoreApplication::translate("InputRZForm", "150", nullptr));
        QTableWidgetItem *___qtablewidgetitem255 = sloteFluTable->verticalHeaderItem(150);
        ___qtablewidgetitem255->setText(QCoreApplication::translate("InputRZForm", "151", nullptr));
        QTableWidgetItem *___qtablewidgetitem256 = sloteFluTable->verticalHeaderItem(151);
        ___qtablewidgetitem256->setText(QCoreApplication::translate("InputRZForm", "152", nullptr));
        QTableWidgetItem *___qtablewidgetitem257 = sloteFluTable->verticalHeaderItem(152);
        ___qtablewidgetitem257->setText(QCoreApplication::translate("InputRZForm", "153", nullptr));
        QTableWidgetItem *___qtablewidgetitem258 = sloteFluTable->verticalHeaderItem(153);
        ___qtablewidgetitem258->setText(QCoreApplication::translate("InputRZForm", "154", nullptr));
        QTableWidgetItem *___qtablewidgetitem259 = sloteFluTable->verticalHeaderItem(154);
        ___qtablewidgetitem259->setText(QCoreApplication::translate("InputRZForm", "155", nullptr));
        QTableWidgetItem *___qtablewidgetitem260 = sloteFluTable->verticalHeaderItem(155);
        ___qtablewidgetitem260->setText(QCoreApplication::translate("InputRZForm", "156", nullptr));
        QTableWidgetItem *___qtablewidgetitem261 = sloteFluTable->verticalHeaderItem(156);
        ___qtablewidgetitem261->setText(QCoreApplication::translate("InputRZForm", "157", nullptr));
        QTableWidgetItem *___qtablewidgetitem262 = sloteFluTable->verticalHeaderItem(157);
        ___qtablewidgetitem262->setText(QCoreApplication::translate("InputRZForm", "158", nullptr));
        QTableWidgetItem *___qtablewidgetitem263 = sloteFluTable->verticalHeaderItem(158);
        ___qtablewidgetitem263->setText(QCoreApplication::translate("InputRZForm", "159", nullptr));
        QTableWidgetItem *___qtablewidgetitem264 = sloteFluTable->verticalHeaderItem(159);
        ___qtablewidgetitem264->setText(QCoreApplication::translate("InputRZForm", "160", nullptr));
        QTableWidgetItem *___qtablewidgetitem265 = sloteFluTable->verticalHeaderItem(160);
        ___qtablewidgetitem265->setText(QCoreApplication::translate("InputRZForm", "161", nullptr));
        QTableWidgetItem *___qtablewidgetitem266 = sloteFluTable->verticalHeaderItem(161);
        ___qtablewidgetitem266->setText(QCoreApplication::translate("InputRZForm", "162", nullptr));
        QTableWidgetItem *___qtablewidgetitem267 = sloteFluTable->verticalHeaderItem(162);
        ___qtablewidgetitem267->setText(QCoreApplication::translate("InputRZForm", "163", nullptr));
        QTableWidgetItem *___qtablewidgetitem268 = sloteFluTable->verticalHeaderItem(163);
        ___qtablewidgetitem268->setText(QCoreApplication::translate("InputRZForm", "164", nullptr));
        QTableWidgetItem *___qtablewidgetitem269 = sloteFluTable->verticalHeaderItem(164);
        ___qtablewidgetitem269->setText(QCoreApplication::translate("InputRZForm", "165", nullptr));
        QTableWidgetItem *___qtablewidgetitem270 = sloteFluTable->verticalHeaderItem(165);
        ___qtablewidgetitem270->setText(QCoreApplication::translate("InputRZForm", "166", nullptr));
        QTableWidgetItem *___qtablewidgetitem271 = sloteFluTable->verticalHeaderItem(166);
        ___qtablewidgetitem271->setText(QCoreApplication::translate("InputRZForm", "167", nullptr));
        QTableWidgetItem *___qtablewidgetitem272 = sloteFluTable->verticalHeaderItem(167);
        ___qtablewidgetitem272->setText(QCoreApplication::translate("InputRZForm", "168", nullptr));
        QTableWidgetItem *___qtablewidgetitem273 = sloteFluTable->verticalHeaderItem(168);
        ___qtablewidgetitem273->setText(QCoreApplication::translate("InputRZForm", "169", nullptr));
        QTableWidgetItem *___qtablewidgetitem274 = sloteFluTable->verticalHeaderItem(169);
        ___qtablewidgetitem274->setText(QCoreApplication::translate("InputRZForm", "170", nullptr));
        QTableWidgetItem *___qtablewidgetitem275 = sloteFluTable->verticalHeaderItem(170);
        ___qtablewidgetitem275->setText(QCoreApplication::translate("InputRZForm", "171", nullptr));
        QTableWidgetItem *___qtablewidgetitem276 = sloteFluTable->verticalHeaderItem(171);
        ___qtablewidgetitem276->setText(QCoreApplication::translate("InputRZForm", "172", nullptr));
        QTableWidgetItem *___qtablewidgetitem277 = sloteFluTable->verticalHeaderItem(172);
        ___qtablewidgetitem277->setText(QCoreApplication::translate("InputRZForm", "173", nullptr));
        QTableWidgetItem *___qtablewidgetitem278 = sloteFluTable->verticalHeaderItem(173);
        ___qtablewidgetitem278->setText(QCoreApplication::translate("InputRZForm", "174", nullptr));
        QTableWidgetItem *___qtablewidgetitem279 = sloteFluTable->verticalHeaderItem(174);
        ___qtablewidgetitem279->setText(QCoreApplication::translate("InputRZForm", "175", nullptr));
        QTableWidgetItem *___qtablewidgetitem280 = sloteFluTable->verticalHeaderItem(175);
        ___qtablewidgetitem280->setText(QCoreApplication::translate("InputRZForm", "176", nullptr));
        QTableWidgetItem *___qtablewidgetitem281 = sloteFluTable->verticalHeaderItem(176);
        ___qtablewidgetitem281->setText(QCoreApplication::translate("InputRZForm", "177", nullptr));
        QTableWidgetItem *___qtablewidgetitem282 = sloteFluTable->verticalHeaderItem(177);
        ___qtablewidgetitem282->setText(QCoreApplication::translate("InputRZForm", "178", nullptr));
        QTableWidgetItem *___qtablewidgetitem283 = sloteFluTable->verticalHeaderItem(178);
        ___qtablewidgetitem283->setText(QCoreApplication::translate("InputRZForm", "179", nullptr));
        QTableWidgetItem *___qtablewidgetitem284 = sloteFluTable->verticalHeaderItem(179);
        ___qtablewidgetitem284->setText(QCoreApplication::translate("InputRZForm", "180", nullptr));
        QTableWidgetItem *___qtablewidgetitem285 = sloteFluTable->verticalHeaderItem(180);
        ___qtablewidgetitem285->setText(QCoreApplication::translate("InputRZForm", "181", nullptr));
        QTableWidgetItem *___qtablewidgetitem286 = sloteFluTable->verticalHeaderItem(181);
        ___qtablewidgetitem286->setText(QCoreApplication::translate("InputRZForm", "182", nullptr));
        QTableWidgetItem *___qtablewidgetitem287 = sloteFluTable->verticalHeaderItem(182);
        ___qtablewidgetitem287->setText(QCoreApplication::translate("InputRZForm", "183", nullptr));
        QTableWidgetItem *___qtablewidgetitem288 = sloteFluTable->verticalHeaderItem(183);
        ___qtablewidgetitem288->setText(QCoreApplication::translate("InputRZForm", "184", nullptr));
        QTableWidgetItem *___qtablewidgetitem289 = sloteFluTable->verticalHeaderItem(184);
        ___qtablewidgetitem289->setText(QCoreApplication::translate("InputRZForm", "185", nullptr));
        QTableWidgetItem *___qtablewidgetitem290 = sloteFluTable->verticalHeaderItem(185);
        ___qtablewidgetitem290->setText(QCoreApplication::translate("InputRZForm", "186", nullptr));
        QTableWidgetItem *___qtablewidgetitem291 = sloteFluTable->verticalHeaderItem(186);
        ___qtablewidgetitem291->setText(QCoreApplication::translate("InputRZForm", "187", nullptr));
        QTableWidgetItem *___qtablewidgetitem292 = sloteFluTable->verticalHeaderItem(187);
        ___qtablewidgetitem292->setText(QCoreApplication::translate("InputRZForm", "188", nullptr));
        QTableWidgetItem *___qtablewidgetitem293 = sloteFluTable->verticalHeaderItem(188);
        ___qtablewidgetitem293->setText(QCoreApplication::translate("InputRZForm", "189", nullptr));
        QTableWidgetItem *___qtablewidgetitem294 = sloteFluTable->verticalHeaderItem(189);
        ___qtablewidgetitem294->setText(QCoreApplication::translate("InputRZForm", "190", nullptr));
        QTableWidgetItem *___qtablewidgetitem295 = sloteFluTable->verticalHeaderItem(190);
        ___qtablewidgetitem295->setText(QCoreApplication::translate("InputRZForm", "191", nullptr));
        QTableWidgetItem *___qtablewidgetitem296 = sloteFluTable->verticalHeaderItem(191);
        ___qtablewidgetitem296->setText(QCoreApplication::translate("InputRZForm", "192", nullptr));
        QTableWidgetItem *___qtablewidgetitem297 = sloteFluTable->verticalHeaderItem(192);
        ___qtablewidgetitem297->setText(QCoreApplication::translate("InputRZForm", "193", nullptr));
        QTableWidgetItem *___qtablewidgetitem298 = sloteFluTable->verticalHeaderItem(193);
        ___qtablewidgetitem298->setText(QCoreApplication::translate("InputRZForm", "194", nullptr));
        QTableWidgetItem *___qtablewidgetitem299 = sloteFluTable->verticalHeaderItem(194);
        ___qtablewidgetitem299->setText(QCoreApplication::translate("InputRZForm", "195", nullptr));
        QTableWidgetItem *___qtablewidgetitem300 = sloteFluTable->verticalHeaderItem(195);
        ___qtablewidgetitem300->setText(QCoreApplication::translate("InputRZForm", "196", nullptr));
        QTableWidgetItem *___qtablewidgetitem301 = sloteFluTable->verticalHeaderItem(196);
        ___qtablewidgetitem301->setText(QCoreApplication::translate("InputRZForm", "197", nullptr));
        QTableWidgetItem *___qtablewidgetitem302 = sloteFluTable->verticalHeaderItem(197);
        ___qtablewidgetitem302->setText(QCoreApplication::translate("InputRZForm", "198", nullptr));
        QTableWidgetItem *___qtablewidgetitem303 = sloteFluTable->verticalHeaderItem(198);
        ___qtablewidgetitem303->setText(QCoreApplication::translate("InputRZForm", "199", nullptr));
        QTableWidgetItem *___qtablewidgetitem304 = sloteFluTable->verticalHeaderItem(199);
        ___qtablewidgetitem304->setText(QCoreApplication::translate("InputRZForm", "200", nullptr));
        TabWidgetRZ->setTabText(TabWidgetRZ->indexOf(IOtab), QCoreApplication::translate("InputRZForm", "I/O control", nullptr));
#if QT_CONFIG(tooltip)
        ncaseLabel->setToolTip(QCoreApplication::translate("InputRZForm", "Histories to run, defaults to 20000", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        ncaseLabel->setWhatsThis(QCoreApplication::translate("InputRZForm", "Histories to run, defaults to 20000", nullptr));
#endif // QT_CONFIG(whatsthis)
        ncaseLabel->setText(QCoreApplication::translate("InputRZForm", "number of histories:", nullptr));
#if QT_CONFIG(tooltip)
        ncaseEdit->setToolTip(QCoreApplication::translate("InputRZForm", "Histories to run, defaults to 20000", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        ncaseEdit->setWhatsThis(QCoreApplication::translate("InputRZForm", "Histories to run, defaults to 20000", nullptr));
#endif // QT_CONFIG(whatsthis)
        ncaseEdit->setText(QCoreApplication::translate("InputRZForm", "20000", nullptr));
#if QT_CONFIG(tooltip)
        maxCPULabel->setToolTip(QCoreApplication::translate("InputRZForm", "terminate cleanly as long as one batch has completed", nullptr));
#endif // QT_CONFIG(tooltip)
        maxCPULabel->setText(QCoreApplication::translate("InputRZForm", "maximum CPU hours allowed:", nullptr));
#if QT_CONFIG(tooltip)
        maxCPUEdit->setToolTip(QCoreApplication::translate("InputRZForm", "terminate cleanly as long as one batch has completed", nullptr));
#endif // QT_CONFIG(tooltip)
        maxCPUEdit->setText(QCoreApplication::translate("InputRZForm", "60", nullptr));
#if QT_CONFIG(tooltip)
        statLabel->setToolTip(QCoreApplication::translate("InputRZForm", "Finish the run if this accuracy is obtained at the end of a batch prior to the total number of histories being run.", nullptr));
#endif // QT_CONFIG(tooltip)
        statLabel->setText(QCoreApplication::translate("InputRZForm", "statistical accuracy sought:", nullptr));
#if QT_CONFIG(tooltip)
        statEdit->setToolTip(QCoreApplication::translate("InputRZForm", "Finish the run if this accuracy is obtained at the end of a batch prior to the total number of histories being run.", nullptr));
#endif // QT_CONFIG(tooltip)
        statEdit->setText(QCoreApplication::translate("InputRZForm", "0.1", nullptr));
        kermaGroupBox->setTitle(QString());
#if QT_CONFIG(tooltip)
        kermaCheckBox->setToolTip(QCoreApplication::translate("InputRZForm", "Score kerma and ratio kerma to dose", nullptr));
#endif // QT_CONFIG(tooltip)
        kermaCheckBox->setText(QCoreApplication::translate("InputRZForm", "score kerma", nullptr));
#if QT_CONFIG(tooltip)
        randGroupBox->setToolTip(QCoreApplication::translate("InputRZForm", "Which generator used is selected in the configuration file.", nullptr));
#endif // QT_CONFIG(tooltip)
        randGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Random Number Initialization", nullptr));
        rand1Label->setText(QCoreApplication::translate("InputRZForm", " 1st.", nullptr));
        rand2Label->setText(QCoreApplication::translate("InputRZForm", "2nd:", nullptr));
        photregGroupBox->setTitle(QString());
#if QT_CONFIG(tooltip)
        photregCheckBox->setToolTip(QCoreApplication::translate("InputRZForm", "regeneration of parent photon after an interaction", nullptr));
#endif // QT_CONFIG(tooltip)
        photregCheckBox->setText(QCoreApplication::translate("InputRZForm", "photon regeneration ", nullptr));
        ifullGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Type of calculation", nullptr));
#if QT_CONFIG(tooltip)
        ifullLabel->setToolTip(QCoreApplication::translate("InputRZForm", "Quantities to score.", nullptr));
#endif // QT_CONFIG(tooltip)
        ifullLabel->setText(QCoreApplication::translate("InputRZForm", " ifull:", nullptr));
        ifullComboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "dose and stoppers", nullptr));
        ifullComboBox->setItemText(1, QCoreApplication::translate("InputRZForm", "Aatt and Ascat", nullptr));
        ifullComboBox->setItemText(2, QCoreApplication::translate("InputRZForm", "Ap", nullptr));
        ifullComboBox->setItemText(3, QCoreApplication::translate("InputRZForm", "Afl and <s>g/w", nullptr));

#if QT_CONFIG(tooltip)
        ifullComboBox->setToolTip(QCoreApplication::translate("InputRZForm", "Quantities to score", nullptr));
#endif // QT_CONFIG(tooltip)
        phdGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Pulse height distribution inputs", nullptr));
        SLOTELabel->setText(QCoreApplication::translate("InputRZForm", "SLOTE :", nullptr));
#if QT_CONFIG(tooltip)
        SLOTEEdit->setToolTip(QCoreApplication::translate("InputRZForm", "Defines output energy bins. If less than zero, input tops of individual energy bins in table below", nullptr));
#endif // QT_CONFIG(tooltip)
        DELTAELabel->setText(QCoreApplication::translate("InputRZForm", "DELTAE :", nullptr));
        QTableWidgetItem *___qtablewidgetitem305 = phdTable->horizontalHeaderItem(0);
        ___qtablewidgetitem305->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem306 = phdTable->horizontalHeaderItem(1);
        ___qtablewidgetitem306->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem307 = phdTable->verticalHeaderItem(0);
        ___qtablewidgetitem307->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem308 = phdTable->verticalHeaderItem(1);
        ___qtablewidgetitem308->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem309 = phdTable->verticalHeaderItem(2);
        ___qtablewidgetitem309->setText(QCoreApplication::translate("InputRZForm", "3", nullptr));
        QTableWidgetItem *___qtablewidgetitem310 = phdTable->verticalHeaderItem(3);
        ___qtablewidgetitem310->setText(QCoreApplication::translate("InputRZForm", "4", nullptr));
        QTableWidgetItem *___qtablewidgetitem311 = phdTable->verticalHeaderItem(4);
        ___qtablewidgetitem311->setText(QCoreApplication::translate("InputRZForm", "5", nullptr));
        QTableWidgetItem *___qtablewidgetitem312 = phdTable->verticalHeaderItem(5);
        ___qtablewidgetitem312->setText(QCoreApplication::translate("InputRZForm", "6", nullptr));
        QTableWidgetItem *___qtablewidgetitem313 = phdTable->verticalHeaderItem(6);
        ___qtablewidgetitem313->setText(QCoreApplication::translate("InputRZForm", "7", nullptr));
        QTableWidgetItem *___qtablewidgetitem314 = phdTable->verticalHeaderItem(7);
        ___qtablewidgetitem314->setText(QCoreApplication::translate("InputRZForm", "8", nullptr));
        QTableWidgetItem *___qtablewidgetitem315 = phdTable->verticalHeaderItem(8);
        ___qtablewidgetitem315->setText(QCoreApplication::translate("InputRZForm", "9", nullptr));
        QTableWidgetItem *___qtablewidgetitem316 = phdTable->verticalHeaderItem(9);
        ___qtablewidgetitem316->setText(QCoreApplication::translate("InputRZForm", "10", nullptr));
        QTableWidgetItem *___qtablewidgetitem317 = phdTable->verticalHeaderItem(10);
        ___qtablewidgetitem317->setText(QCoreApplication::translate("InputRZForm", "11", nullptr));
        QTableWidgetItem *___qtablewidgetitem318 = phdTable->verticalHeaderItem(11);
        ___qtablewidgetitem318->setText(QCoreApplication::translate("InputRZForm", "12", nullptr));
        QTableWidgetItem *___qtablewidgetitem319 = phdTable->verticalHeaderItem(12);
        ___qtablewidgetitem319->setText(QCoreApplication::translate("InputRZForm", "13", nullptr));
        QTableWidgetItem *___qtablewidgetitem320 = phdTable->verticalHeaderItem(13);
        ___qtablewidgetitem320->setText(QCoreApplication::translate("InputRZForm", "14", nullptr));
        QTableWidgetItem *___qtablewidgetitem321 = phdTable->verticalHeaderItem(14);
        ___qtablewidgetitem321->setText(QCoreApplication::translate("InputRZForm", "15", nullptr));
        QTableWidgetItem *___qtablewidgetitem322 = phdTable->verticalHeaderItem(15);
        ___qtablewidgetitem322->setText(QCoreApplication::translate("InputRZForm", "16", nullptr));
        QTableWidgetItem *___qtablewidgetitem323 = phdTable->verticalHeaderItem(16);
        ___qtablewidgetitem323->setText(QCoreApplication::translate("InputRZForm", "17", nullptr));
        QTableWidgetItem *___qtablewidgetitem324 = phdTable->verticalHeaderItem(17);
        ___qtablewidgetitem324->setText(QCoreApplication::translate("InputRZForm", "18", nullptr));
        QTableWidgetItem *___qtablewidgetitem325 = phdTable->verticalHeaderItem(18);
        ___qtablewidgetitem325->setText(QCoreApplication::translate("InputRZForm", "19", nullptr));
        QTableWidgetItem *___qtablewidgetitem326 = phdTable->verticalHeaderItem(19);
        ___qtablewidgetitem326->setText(QCoreApplication::translate("InputRZForm", "20", nullptr));
        QTableWidgetItem *___qtablewidgetitem327 = phdTable->verticalHeaderItem(20);
        ___qtablewidgetitem327->setText(QCoreApplication::translate("InputRZForm", "21", nullptr));
        QTableWidgetItem *___qtablewidgetitem328 = phdTable->verticalHeaderItem(21);
        ___qtablewidgetitem328->setText(QCoreApplication::translate("InputRZForm", "22", nullptr));
        QTableWidgetItem *___qtablewidgetitem329 = phdTable->verticalHeaderItem(22);
        ___qtablewidgetitem329->setText(QCoreApplication::translate("InputRZForm", "23", nullptr));
        QTableWidgetItem *___qtablewidgetitem330 = phdTable->verticalHeaderItem(23);
        ___qtablewidgetitem330->setText(QCoreApplication::translate("InputRZForm", "24", nullptr));
        QTableWidgetItem *___qtablewidgetitem331 = phdTable->verticalHeaderItem(24);
        ___qtablewidgetitem331->setText(QCoreApplication::translate("InputRZForm", "25", nullptr));
        QTableWidgetItem *___qtablewidgetitem332 = phdTable->verticalHeaderItem(25);
        ___qtablewidgetitem332->setText(QCoreApplication::translate("InputRZForm", "26", nullptr));
        QTableWidgetItem *___qtablewidgetitem333 = phdTable->verticalHeaderItem(26);
        ___qtablewidgetitem333->setText(QCoreApplication::translate("InputRZForm", "27", nullptr));
        QTableWidgetItem *___qtablewidgetitem334 = phdTable->verticalHeaderItem(27);
        ___qtablewidgetitem334->setText(QCoreApplication::translate("InputRZForm", "28", nullptr));
        QTableWidgetItem *___qtablewidgetitem335 = phdTable->verticalHeaderItem(28);
        ___qtablewidgetitem335->setText(QCoreApplication::translate("InputRZForm", "29", nullptr));
        QTableWidgetItem *___qtablewidgetitem336 = phdTable->verticalHeaderItem(29);
        ___qtablewidgetitem336->setText(QCoreApplication::translate("InputRZForm", "30", nullptr));
        QTableWidgetItem *___qtablewidgetitem337 = phdTable->verticalHeaderItem(30);
        ___qtablewidgetitem337->setText(QCoreApplication::translate("InputRZForm", "31", nullptr));
        QTableWidgetItem *___qtablewidgetitem338 = phdTable->verticalHeaderItem(31);
        ___qtablewidgetitem338->setText(QCoreApplication::translate("InputRZForm", "32", nullptr));
        QTableWidgetItem *___qtablewidgetitem339 = phdTable->verticalHeaderItem(32);
        ___qtablewidgetitem339->setText(QCoreApplication::translate("InputRZForm", "33", nullptr));
        QTableWidgetItem *___qtablewidgetitem340 = phdTable->verticalHeaderItem(33);
        ___qtablewidgetitem340->setText(QCoreApplication::translate("InputRZForm", "34", nullptr));
        QTableWidgetItem *___qtablewidgetitem341 = phdTable->verticalHeaderItem(34);
        ___qtablewidgetitem341->setText(QCoreApplication::translate("InputRZForm", "35", nullptr));
        QTableWidgetItem *___qtablewidgetitem342 = phdTable->verticalHeaderItem(35);
        ___qtablewidgetitem342->setText(QCoreApplication::translate("InputRZForm", "36", nullptr));
        QTableWidgetItem *___qtablewidgetitem343 = phdTable->verticalHeaderItem(36);
        ___qtablewidgetitem343->setText(QCoreApplication::translate("InputRZForm", "37", nullptr));
        QTableWidgetItem *___qtablewidgetitem344 = phdTable->verticalHeaderItem(37);
        ___qtablewidgetitem344->setText(QCoreApplication::translate("InputRZForm", "38", nullptr));
        QTableWidgetItem *___qtablewidgetitem345 = phdTable->verticalHeaderItem(38);
        ___qtablewidgetitem345->setText(QCoreApplication::translate("InputRZForm", "39", nullptr));
        QTableWidgetItem *___qtablewidgetitem346 = phdTable->verticalHeaderItem(39);
        ___qtablewidgetitem346->setText(QCoreApplication::translate("InputRZForm", "40", nullptr));
        QTableWidgetItem *___qtablewidgetitem347 = phdTable->verticalHeaderItem(40);
        ___qtablewidgetitem347->setText(QCoreApplication::translate("InputRZForm", "41", nullptr));
        QTableWidgetItem *___qtablewidgetitem348 = phdTable->verticalHeaderItem(41);
        ___qtablewidgetitem348->setText(QCoreApplication::translate("InputRZForm", "42", nullptr));
        QTableWidgetItem *___qtablewidgetitem349 = phdTable->verticalHeaderItem(42);
        ___qtablewidgetitem349->setText(QCoreApplication::translate("InputRZForm", "43", nullptr));
        QTableWidgetItem *___qtablewidgetitem350 = phdTable->verticalHeaderItem(43);
        ___qtablewidgetitem350->setText(QCoreApplication::translate("InputRZForm", "44", nullptr));
        QTableWidgetItem *___qtablewidgetitem351 = phdTable->verticalHeaderItem(44);
        ___qtablewidgetitem351->setText(QCoreApplication::translate("InputRZForm", "45", nullptr));
        QTableWidgetItem *___qtablewidgetitem352 = phdTable->verticalHeaderItem(45);
        ___qtablewidgetitem352->setText(QCoreApplication::translate("InputRZForm", "46", nullptr));
        QTableWidgetItem *___qtablewidgetitem353 = phdTable->verticalHeaderItem(46);
        ___qtablewidgetitem353->setText(QCoreApplication::translate("InputRZForm", "47", nullptr));
        QTableWidgetItem *___qtablewidgetitem354 = phdTable->verticalHeaderItem(47);
        ___qtablewidgetitem354->setText(QCoreApplication::translate("InputRZForm", "48", nullptr));
        QTableWidgetItem *___qtablewidgetitem355 = phdTable->verticalHeaderItem(48);
        ___qtablewidgetitem355->setText(QCoreApplication::translate("InputRZForm", "49", nullptr));
        QTableWidgetItem *___qtablewidgetitem356 = phdTable->verticalHeaderItem(49);
        ___qtablewidgetitem356->setText(QCoreApplication::translate("InputRZForm", "50", nullptr));
        TabWidgetRZ->setTabText(TabWidgetRZ->indexOf(MCtab), QCoreApplication::translate("InputRZForm", "Monte Carlo", nullptr));
        inputmethButtonGroup->setTitle(QCoreApplication::translate("InputRZForm", "input method", nullptr));
        groupRadioButton->setText(QCoreApplication::translate("InputRZForm", "groups", nullptr));
        individualRadioButton->setText(QCoreApplication::translate("InputRZForm", "individual", nullptr));
        cavityRadioButton->setText(QCoreApplication::translate("InputRZForm", "cavity description", nullptr));
#if QT_CONFIG(tooltip)
        ZFaceLabel->setToolTip(QCoreApplication::translate("InputRZForm", "Start of first slab.", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        ZFaceLabel->setWhatsThis(QCoreApplication::translate("InputRZForm", "Start of first slab.", nullptr));
#endif // QT_CONFIG(whatsthis)
        ZFaceLabel->setText(QCoreApplication::translate("InputRZForm", "Z of front face:", nullptr));
#if QT_CONFIG(tooltip)
        ZFaceEdit->setToolTip(QCoreApplication::translate("InputRZForm", "Start of first slab.", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        ZFaceEdit->setWhatsThis(QCoreApplication::translate("InputRZForm", "Start of first slab.", nullptr));
#endif // QT_CONFIG(whatsthis)
        ZFaceEdit->setText(QCoreApplication::translate("InputRZForm", "0", nullptr));
        CavityInfoLabel->setText(QCoreApplication::translate("InputRZForm", "<=== See cavity specification on next Tab", nullptr));
        mediaGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Media input", nullptr));
        mediaLabel->setText(QCoreApplication::translate("InputRZForm", "description by", nullptr));
        mediaComboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "regions", nullptr));
        mediaComboBox->setItemText(1, QCoreApplication::translate("InputRZForm", "planes", nullptr));

        QTableWidgetItem *___qtablewidgetitem357 = mediaTable->horizontalHeaderItem(0);
        ___qtablewidgetitem357->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem358 = mediaTable->horizontalHeaderItem(1);
        ___qtablewidgetitem358->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem359 = mediaTable->horizontalHeaderItem(2);
        ___qtablewidgetitem359->setText(QCoreApplication::translate("InputRZForm", "3", nullptr));
        QTableWidgetItem *___qtablewidgetitem360 = mediaTable->verticalHeaderItem(0);
        ___qtablewidgetitem360->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem361 = mediaTable->verticalHeaderItem(1);
        ___qtablewidgetitem361->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem362 = mediaTable->verticalHeaderItem(2);
        ___qtablewidgetitem362->setText(QCoreApplication::translate("InputRZForm", "3", nullptr));
        QTableWidgetItem *___qtablewidgetitem363 = mediaTable->verticalHeaderItem(3);
        ___qtablewidgetitem363->setText(QCoreApplication::translate("InputRZForm", "4", nullptr));
        QTableWidgetItem *___qtablewidgetitem364 = mediaTable->verticalHeaderItem(4);
        ___qtablewidgetitem364->setText(QCoreApplication::translate("InputRZForm", "5", nullptr));
        QTableWidgetItem *___qtablewidgetitem365 = mediaTable->verticalHeaderItem(5);
        ___qtablewidgetitem365->setText(QCoreApplication::translate("InputRZForm", "6", nullptr));
        QTableWidgetItem *___qtablewidgetitem366 = mediaTable->verticalHeaderItem(6);
        ___qtablewidgetitem366->setText(QCoreApplication::translate("InputRZForm", "7", nullptr));
        QTableWidgetItem *___qtablewidgetitem367 = mediaTable->verticalHeaderItem(7);
        ___qtablewidgetitem367->setText(QCoreApplication::translate("InputRZForm", "8", nullptr));
        QTableWidgetItem *___qtablewidgetitem368 = mediaTable->verticalHeaderItem(8);
        ___qtablewidgetitem368->setText(QCoreApplication::translate("InputRZForm", "9", nullptr));
        QTableWidgetItem *___qtablewidgetitem369 = mediaTable->verticalHeaderItem(9);
        ___qtablewidgetitem369->setText(QCoreApplication::translate("InputRZForm", "10", nullptr));
        QTableWidgetItem *___qtablewidgetitem370 = mediaTable->verticalHeaderItem(10);
        ___qtablewidgetitem370->setText(QCoreApplication::translate("InputRZForm", "11", nullptr));
        QTableWidgetItem *___qtablewidgetitem371 = mediaTable->verticalHeaderItem(11);
        ___qtablewidgetitem371->setText(QCoreApplication::translate("InputRZForm", "12", nullptr));
        QTableWidgetItem *___qtablewidgetitem372 = mediaTable->verticalHeaderItem(12);
        ___qtablewidgetitem372->setText(QCoreApplication::translate("InputRZForm", "13", nullptr));
        QTableWidgetItem *___qtablewidgetitem373 = mediaTable->verticalHeaderItem(13);
        ___qtablewidgetitem373->setText(QCoreApplication::translate("InputRZForm", "14", nullptr));
        QTableWidgetItem *___qtablewidgetitem374 = mediaTable->verticalHeaderItem(14);
        ___qtablewidgetitem374->setText(QCoreApplication::translate("InputRZForm", "15", nullptr));
        QTableWidgetItem *___qtablewidgetitem375 = mediaTable->verticalHeaderItem(15);
        ___qtablewidgetitem375->setText(QCoreApplication::translate("InputRZForm", "16", nullptr));
        QTableWidgetItem *___qtablewidgetitem376 = mediaTable->verticalHeaderItem(16);
        ___qtablewidgetitem376->setText(QCoreApplication::translate("InputRZForm", "17", nullptr));
        QTableWidgetItem *___qtablewidgetitem377 = mediaTable->verticalHeaderItem(17);
        ___qtablewidgetitem377->setText(QCoreApplication::translate("InputRZForm", "18", nullptr));
        QTableWidgetItem *___qtablewidgetitem378 = mediaTable->verticalHeaderItem(18);
        ___qtablewidgetitem378->setText(QCoreApplication::translate("InputRZForm", "19", nullptr));
        QTableWidgetItem *___qtablewidgetitem379 = mediaTable->verticalHeaderItem(19);
        ___qtablewidgetitem379->setText(QCoreApplication::translate("InputRZForm", "20", nullptr));
        QTableWidgetItem *___qtablewidgetitem380 = mediaTable->verticalHeaderItem(20);
        ___qtablewidgetitem380->setText(QCoreApplication::translate("InputRZForm", "21", nullptr));
        QTableWidgetItem *___qtablewidgetitem381 = mediaTable->verticalHeaderItem(21);
        ___qtablewidgetitem381->setText(QCoreApplication::translate("InputRZForm", "22", nullptr));
        QTableWidgetItem *___qtablewidgetitem382 = mediaTable->verticalHeaderItem(22);
        ___qtablewidgetitem382->setText(QCoreApplication::translate("InputRZForm", "23", nullptr));
        QTableWidgetItem *___qtablewidgetitem383 = mediaTable->verticalHeaderItem(23);
        ___qtablewidgetitem383->setText(QCoreApplication::translate("InputRZForm", "24", nullptr));
        QTableWidgetItem *___qtablewidgetitem384 = mediaTable->verticalHeaderItem(24);
        ___qtablewidgetitem384->setText(QCoreApplication::translate("InputRZForm", "25", nullptr));
        QTableWidgetItem *___qtablewidgetitem385 = mediaTable->verticalHeaderItem(25);
        ___qtablewidgetitem385->setText(QCoreApplication::translate("InputRZForm", "26", nullptr));
        QTableWidgetItem *___qtablewidgetitem386 = mediaTable->verticalHeaderItem(26);
        ___qtablewidgetitem386->setText(QCoreApplication::translate("InputRZForm", "27", nullptr));
        QTableWidgetItem *___qtablewidgetitem387 = mediaTable->verticalHeaderItem(27);
        ___qtablewidgetitem387->setText(QCoreApplication::translate("InputRZForm", "28", nullptr));
        QTableWidgetItem *___qtablewidgetitem388 = mediaTable->verticalHeaderItem(28);
        ___qtablewidgetitem388->setText(QCoreApplication::translate("InputRZForm", "29", nullptr));
        QTableWidgetItem *___qtablewidgetitem389 = mediaTable->verticalHeaderItem(29);
        ___qtablewidgetitem389->setText(QCoreApplication::translate("InputRZForm", "30", nullptr));
        QTableWidgetItem *___qtablewidgetitem390 = mediaTable->verticalHeaderItem(30);
        ___qtablewidgetitem390->setText(QCoreApplication::translate("InputRZForm", "31", nullptr));
        QTableWidgetItem *___qtablewidgetitem391 = mediaTable->verticalHeaderItem(31);
        ___qtablewidgetitem391->setText(QCoreApplication::translate("InputRZForm", "32", nullptr));
        QTableWidgetItem *___qtablewidgetitem392 = mediaTable->verticalHeaderItem(32);
        ___qtablewidgetitem392->setText(QCoreApplication::translate("InputRZForm", "33", nullptr));
        QTableWidgetItem *___qtablewidgetitem393 = mediaTable->verticalHeaderItem(33);
        ___qtablewidgetitem393->setText(QCoreApplication::translate("InputRZForm", "34", nullptr));
        QTableWidgetItem *___qtablewidgetitem394 = mediaTable->verticalHeaderItem(34);
        ___qtablewidgetitem394->setText(QCoreApplication::translate("InputRZForm", "35", nullptr));
        QTableWidgetItem *___qtablewidgetitem395 = mediaTable->verticalHeaderItem(35);
        ___qtablewidgetitem395->setText(QCoreApplication::translate("InputRZForm", "36", nullptr));
        QTableWidgetItem *___qtablewidgetitem396 = mediaTable->verticalHeaderItem(36);
        ___qtablewidgetitem396->setText(QCoreApplication::translate("InputRZForm", "37", nullptr));
        QTableWidgetItem *___qtablewidgetitem397 = mediaTable->verticalHeaderItem(37);
        ___qtablewidgetitem397->setText(QCoreApplication::translate("InputRZForm", "38", nullptr));
        QTableWidgetItem *___qtablewidgetitem398 = mediaTable->verticalHeaderItem(38);
        ___qtablewidgetitem398->setText(QCoreApplication::translate("InputRZForm", "39", nullptr));
        QTableWidgetItem *___qtablewidgetitem399 = mediaTable->verticalHeaderItem(39);
        ___qtablewidgetitem399->setText(QCoreApplication::translate("InputRZForm", "40", nullptr));
        QTableWidgetItem *___qtablewidgetitem400 = mediaTable->verticalHeaderItem(40);
        ___qtablewidgetitem400->setText(QCoreApplication::translate("InputRZForm", "41", nullptr));
        QTableWidgetItem *___qtablewidgetitem401 = mediaTable->verticalHeaderItem(41);
        ___qtablewidgetitem401->setText(QCoreApplication::translate("InputRZForm", "42", nullptr));
        QTableWidgetItem *___qtablewidgetitem402 = mediaTable->verticalHeaderItem(42);
        ___qtablewidgetitem402->setText(QCoreApplication::translate("InputRZForm", "43", nullptr));
        QTableWidgetItem *___qtablewidgetitem403 = mediaTable->verticalHeaderItem(43);
        ___qtablewidgetitem403->setText(QCoreApplication::translate("InputRZForm", "44", nullptr));
        QTableWidgetItem *___qtablewidgetitem404 = mediaTable->verticalHeaderItem(44);
        ___qtablewidgetitem404->setText(QCoreApplication::translate("InputRZForm", "45", nullptr));
        QTableWidgetItem *___qtablewidgetitem405 = mediaTable->verticalHeaderItem(45);
        ___qtablewidgetitem405->setText(QCoreApplication::translate("InputRZForm", "46", nullptr));
        QTableWidgetItem *___qtablewidgetitem406 = mediaTable->verticalHeaderItem(46);
        ___qtablewidgetitem406->setText(QCoreApplication::translate("InputRZForm", "47", nullptr));
        QTableWidgetItem *___qtablewidgetitem407 = mediaTable->verticalHeaderItem(47);
        ___qtablewidgetitem407->setText(QCoreApplication::translate("InputRZForm", "48", nullptr));
        QTableWidgetItem *___qtablewidgetitem408 = mediaTable->verticalHeaderItem(48);
        ___qtablewidgetitem408->setText(QCoreApplication::translate("InputRZForm", "49", nullptr));
        QTableWidgetItem *___qtablewidgetitem409 = mediaTable->verticalHeaderItem(49);
        ___qtablewidgetitem409->setText(QCoreApplication::translate("InputRZForm", "50", nullptr));
#if QT_CONFIG(tooltip)
        mediaTable->setToolTip(QCoreApplication::translate("InputRZForm", "Assign media to the regions.", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        mediaTable->setWhatsThis(QCoreApplication::translate("InputRZForm", "Assign media to the regions.", nullptr));
#endif // QT_CONFIG(whatsthis)
#if QT_CONFIG(tooltip)
        GroupBox53->setToolTip(QCoreApplication::translate("InputRZForm", "Define planes.", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        GroupBox53->setWhatsThis(QCoreApplication::translate("InputRZForm", "Define planes.", nullptr));
#endif // QT_CONFIG(whatsthis)
        GroupBox53->setTitle(QCoreApplication::translate("InputRZForm", "planes information", nullptr));
        QTableWidgetItem *___qtablewidgetitem410 = geometryTable->horizontalHeaderItem(0);
        ___qtablewidgetitem410->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem411 = geometryTable->horizontalHeaderItem(1);
        ___qtablewidgetitem411->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem412 = geometryTable->verticalHeaderItem(0);
        ___qtablewidgetitem412->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem413 = geometryTable->verticalHeaderItem(1);
        ___qtablewidgetitem413->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem414 = geometryTable->verticalHeaderItem(2);
        ___qtablewidgetitem414->setText(QCoreApplication::translate("InputRZForm", "3", nullptr));
        QTableWidgetItem *___qtablewidgetitem415 = geometryTable->verticalHeaderItem(3);
        ___qtablewidgetitem415->setText(QCoreApplication::translate("InputRZForm", "4", nullptr));
        QTableWidgetItem *___qtablewidgetitem416 = geometryTable->verticalHeaderItem(4);
        ___qtablewidgetitem416->setText(QCoreApplication::translate("InputRZForm", "5", nullptr));
        QTableWidgetItem *___qtablewidgetitem417 = geometryTable->verticalHeaderItem(5);
        ___qtablewidgetitem417->setText(QCoreApplication::translate("InputRZForm", "6", nullptr));
        QTableWidgetItem *___qtablewidgetitem418 = geometryTable->verticalHeaderItem(6);
        ___qtablewidgetitem418->setText(QCoreApplication::translate("InputRZForm", "7", nullptr));
        QTableWidgetItem *___qtablewidgetitem419 = geometryTable->verticalHeaderItem(7);
        ___qtablewidgetitem419->setText(QCoreApplication::translate("InputRZForm", "8", nullptr));
        QTableWidgetItem *___qtablewidgetitem420 = geometryTable->verticalHeaderItem(8);
        ___qtablewidgetitem420->setText(QCoreApplication::translate("InputRZForm", "9", nullptr));
        QTableWidgetItem *___qtablewidgetitem421 = geometryTable->verticalHeaderItem(9);
        ___qtablewidgetitem421->setText(QCoreApplication::translate("InputRZForm", "10", nullptr));
        QTableWidgetItem *___qtablewidgetitem422 = geometryTable->verticalHeaderItem(10);
        ___qtablewidgetitem422->setText(QCoreApplication::translate("InputRZForm", "11", nullptr));
        QTableWidgetItem *___qtablewidgetitem423 = geometryTable->verticalHeaderItem(11);
        ___qtablewidgetitem423->setText(QCoreApplication::translate("InputRZForm", "12", nullptr));
        QTableWidgetItem *___qtablewidgetitem424 = geometryTable->verticalHeaderItem(12);
        ___qtablewidgetitem424->setText(QCoreApplication::translate("InputRZForm", "13", nullptr));
        QTableWidgetItem *___qtablewidgetitem425 = geometryTable->verticalHeaderItem(13);
        ___qtablewidgetitem425->setText(QCoreApplication::translate("InputRZForm", "14", nullptr));
        QTableWidgetItem *___qtablewidgetitem426 = geometryTable->verticalHeaderItem(14);
        ___qtablewidgetitem426->setText(QCoreApplication::translate("InputRZForm", "15", nullptr));
        QTableWidgetItem *___qtablewidgetitem427 = geometryTable->verticalHeaderItem(15);
        ___qtablewidgetitem427->setText(QCoreApplication::translate("InputRZForm", "16", nullptr));
        QTableWidgetItem *___qtablewidgetitem428 = geometryTable->verticalHeaderItem(16);
        ___qtablewidgetitem428->setText(QCoreApplication::translate("InputRZForm", "17", nullptr));
        QTableWidgetItem *___qtablewidgetitem429 = geometryTable->verticalHeaderItem(17);
        ___qtablewidgetitem429->setText(QCoreApplication::translate("InputRZForm", "18", nullptr));
        QTableWidgetItem *___qtablewidgetitem430 = geometryTable->verticalHeaderItem(18);
        ___qtablewidgetitem430->setText(QCoreApplication::translate("InputRZForm", "19", nullptr));
        QTableWidgetItem *___qtablewidgetitem431 = geometryTable->verticalHeaderItem(19);
        ___qtablewidgetitem431->setText(QCoreApplication::translate("InputRZForm", "20", nullptr));
        QTableWidgetItem *___qtablewidgetitem432 = geometryTable->verticalHeaderItem(20);
        ___qtablewidgetitem432->setText(QCoreApplication::translate("InputRZForm", "21", nullptr));
        QTableWidgetItem *___qtablewidgetitem433 = geometryTable->verticalHeaderItem(21);
        ___qtablewidgetitem433->setText(QCoreApplication::translate("InputRZForm", "22", nullptr));
        QTableWidgetItem *___qtablewidgetitem434 = geometryTable->verticalHeaderItem(22);
        ___qtablewidgetitem434->setText(QCoreApplication::translate("InputRZForm", "23", nullptr));
        QTableWidgetItem *___qtablewidgetitem435 = geometryTable->verticalHeaderItem(23);
        ___qtablewidgetitem435->setText(QCoreApplication::translate("InputRZForm", "24", nullptr));
        QTableWidgetItem *___qtablewidgetitem436 = geometryTable->verticalHeaderItem(24);
        ___qtablewidgetitem436->setText(QCoreApplication::translate("InputRZForm", "25", nullptr));
        QTableWidgetItem *___qtablewidgetitem437 = geometryTable->verticalHeaderItem(25);
        ___qtablewidgetitem437->setText(QCoreApplication::translate("InputRZForm", "26", nullptr));
        QTableWidgetItem *___qtablewidgetitem438 = geometryTable->verticalHeaderItem(26);
        ___qtablewidgetitem438->setText(QCoreApplication::translate("InputRZForm", "27", nullptr));
        QTableWidgetItem *___qtablewidgetitem439 = geometryTable->verticalHeaderItem(27);
        ___qtablewidgetitem439->setText(QCoreApplication::translate("InputRZForm", "28", nullptr));
        QTableWidgetItem *___qtablewidgetitem440 = geometryTable->verticalHeaderItem(28);
        ___qtablewidgetitem440->setText(QCoreApplication::translate("InputRZForm", "29", nullptr));
        QTableWidgetItem *___qtablewidgetitem441 = geometryTable->verticalHeaderItem(29);
        ___qtablewidgetitem441->setText(QCoreApplication::translate("InputRZForm", "30", nullptr));
        QTableWidgetItem *___qtablewidgetitem442 = geometryTable->verticalHeaderItem(30);
        ___qtablewidgetitem442->setText(QCoreApplication::translate("InputRZForm", "31", nullptr));
        QTableWidgetItem *___qtablewidgetitem443 = geometryTable->verticalHeaderItem(31);
        ___qtablewidgetitem443->setText(QCoreApplication::translate("InputRZForm", "32", nullptr));
        QTableWidgetItem *___qtablewidgetitem444 = geometryTable->verticalHeaderItem(32);
        ___qtablewidgetitem444->setText(QCoreApplication::translate("InputRZForm", "33", nullptr));
        QTableWidgetItem *___qtablewidgetitem445 = geometryTable->verticalHeaderItem(33);
        ___qtablewidgetitem445->setText(QCoreApplication::translate("InputRZForm", "34", nullptr));
        QTableWidgetItem *___qtablewidgetitem446 = geometryTable->verticalHeaderItem(34);
        ___qtablewidgetitem446->setText(QCoreApplication::translate("InputRZForm", "35", nullptr));
        QTableWidgetItem *___qtablewidgetitem447 = geometryTable->verticalHeaderItem(35);
        ___qtablewidgetitem447->setText(QCoreApplication::translate("InputRZForm", "36", nullptr));
        QTableWidgetItem *___qtablewidgetitem448 = geometryTable->verticalHeaderItem(36);
        ___qtablewidgetitem448->setText(QCoreApplication::translate("InputRZForm", "37", nullptr));
        QTableWidgetItem *___qtablewidgetitem449 = geometryTable->verticalHeaderItem(37);
        ___qtablewidgetitem449->setText(QCoreApplication::translate("InputRZForm", "38", nullptr));
        QTableWidgetItem *___qtablewidgetitem450 = geometryTable->verticalHeaderItem(38);
        ___qtablewidgetitem450->setText(QCoreApplication::translate("InputRZForm", "39", nullptr));
        QTableWidgetItem *___qtablewidgetitem451 = geometryTable->verticalHeaderItem(39);
        ___qtablewidgetitem451->setText(QCoreApplication::translate("InputRZForm", "40", nullptr));
        QTableWidgetItem *___qtablewidgetitem452 = geometryTable->verticalHeaderItem(40);
        ___qtablewidgetitem452->setText(QCoreApplication::translate("InputRZForm", "41", nullptr));
        QTableWidgetItem *___qtablewidgetitem453 = geometryTable->verticalHeaderItem(41);
        ___qtablewidgetitem453->setText(QCoreApplication::translate("InputRZForm", "42", nullptr));
        QTableWidgetItem *___qtablewidgetitem454 = geometryTable->verticalHeaderItem(42);
        ___qtablewidgetitem454->setText(QCoreApplication::translate("InputRZForm", "43", nullptr));
        QTableWidgetItem *___qtablewidgetitem455 = geometryTable->verticalHeaderItem(43);
        ___qtablewidgetitem455->setText(QCoreApplication::translate("InputRZForm", "44", nullptr));
        QTableWidgetItem *___qtablewidgetitem456 = geometryTable->verticalHeaderItem(44);
        ___qtablewidgetitem456->setText(QCoreApplication::translate("InputRZForm", "45", nullptr));
        QTableWidgetItem *___qtablewidgetitem457 = geometryTable->verticalHeaderItem(45);
        ___qtablewidgetitem457->setText(QCoreApplication::translate("InputRZForm", "46", nullptr));
        QTableWidgetItem *___qtablewidgetitem458 = geometryTable->verticalHeaderItem(46);
        ___qtablewidgetitem458->setText(QCoreApplication::translate("InputRZForm", "47", nullptr));
        QTableWidgetItem *___qtablewidgetitem459 = geometryTable->verticalHeaderItem(47);
        ___qtablewidgetitem459->setText(QCoreApplication::translate("InputRZForm", "48", nullptr));
        QTableWidgetItem *___qtablewidgetitem460 = geometryTable->verticalHeaderItem(48);
        ___qtablewidgetitem460->setText(QCoreApplication::translate("InputRZForm", "49", nullptr));
        QTableWidgetItem *___qtablewidgetitem461 = geometryTable->verticalHeaderItem(49);
        ___qtablewidgetitem461->setText(QCoreApplication::translate("InputRZForm", "50", nullptr));
#if QT_CONFIG(tooltip)
        GroupBox54->setToolTip(QCoreApplication::translate("InputRZForm", "Input cylinder radii.", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        GroupBox54->setWhatsThis(QCoreApplication::translate("InputRZForm", "Input cylinder radii.", nullptr));
#endif // QT_CONFIG(whatsthis)
        GroupBox54->setTitle(QCoreApplication::translate("InputRZForm", "cylinder information", nullptr));
        QTableWidgetItem *___qtablewidgetitem462 = cylTable->horizontalHeaderItem(0);
        ___qtablewidgetitem462->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem463 = cylTable->verticalHeaderItem(0);
        ___qtablewidgetitem463->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem464 = cylTable->verticalHeaderItem(1);
        ___qtablewidgetitem464->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem465 = cylTable->verticalHeaderItem(2);
        ___qtablewidgetitem465->setText(QCoreApplication::translate("InputRZForm", "3", nullptr));
        QTableWidgetItem *___qtablewidgetitem466 = cylTable->verticalHeaderItem(3);
        ___qtablewidgetitem466->setText(QCoreApplication::translate("InputRZForm", "4", nullptr));
        QTableWidgetItem *___qtablewidgetitem467 = cylTable->verticalHeaderItem(4);
        ___qtablewidgetitem467->setText(QCoreApplication::translate("InputRZForm", "5", nullptr));
        QTableWidgetItem *___qtablewidgetitem468 = cylTable->verticalHeaderItem(5);
        ___qtablewidgetitem468->setText(QCoreApplication::translate("InputRZForm", "6", nullptr));
        QTableWidgetItem *___qtablewidgetitem469 = cylTable->verticalHeaderItem(6);
        ___qtablewidgetitem469->setText(QCoreApplication::translate("InputRZForm", "7", nullptr));
        QTableWidgetItem *___qtablewidgetitem470 = cylTable->verticalHeaderItem(7);
        ___qtablewidgetitem470->setText(QCoreApplication::translate("InputRZForm", "8", nullptr));
        QTableWidgetItem *___qtablewidgetitem471 = cylTable->verticalHeaderItem(8);
        ___qtablewidgetitem471->setText(QCoreApplication::translate("InputRZForm", "9", nullptr));
        QTableWidgetItem *___qtablewidgetitem472 = cylTable->verticalHeaderItem(9);
        ___qtablewidgetitem472->setText(QCoreApplication::translate("InputRZForm", "10", nullptr));
        QTableWidgetItem *___qtablewidgetitem473 = cylTable->verticalHeaderItem(10);
        ___qtablewidgetitem473->setText(QCoreApplication::translate("InputRZForm", "11", nullptr));
        QTableWidgetItem *___qtablewidgetitem474 = cylTable->verticalHeaderItem(11);
        ___qtablewidgetitem474->setText(QCoreApplication::translate("InputRZForm", "12", nullptr));
        QTableWidgetItem *___qtablewidgetitem475 = cylTable->verticalHeaderItem(12);
        ___qtablewidgetitem475->setText(QCoreApplication::translate("InputRZForm", "13", nullptr));
        QTableWidgetItem *___qtablewidgetitem476 = cylTable->verticalHeaderItem(13);
        ___qtablewidgetitem476->setText(QCoreApplication::translate("InputRZForm", "14", nullptr));
        QTableWidgetItem *___qtablewidgetitem477 = cylTable->verticalHeaderItem(14);
        ___qtablewidgetitem477->setText(QCoreApplication::translate("InputRZForm", "15", nullptr));
        QTableWidgetItem *___qtablewidgetitem478 = cylTable->verticalHeaderItem(15);
        ___qtablewidgetitem478->setText(QCoreApplication::translate("InputRZForm", "16", nullptr));
        QTableWidgetItem *___qtablewidgetitem479 = cylTable->verticalHeaderItem(16);
        ___qtablewidgetitem479->setText(QCoreApplication::translate("InputRZForm", "17", nullptr));
        QTableWidgetItem *___qtablewidgetitem480 = cylTable->verticalHeaderItem(17);
        ___qtablewidgetitem480->setText(QCoreApplication::translate("InputRZForm", "18", nullptr));
        QTableWidgetItem *___qtablewidgetitem481 = cylTable->verticalHeaderItem(18);
        ___qtablewidgetitem481->setText(QCoreApplication::translate("InputRZForm", "19", nullptr));
        QTableWidgetItem *___qtablewidgetitem482 = cylTable->verticalHeaderItem(19);
        ___qtablewidgetitem482->setText(QCoreApplication::translate("InputRZForm", "20", nullptr));
        QTableWidgetItem *___qtablewidgetitem483 = cylTable->verticalHeaderItem(20);
        ___qtablewidgetitem483->setText(QCoreApplication::translate("InputRZForm", "21", nullptr));
        QTableWidgetItem *___qtablewidgetitem484 = cylTable->verticalHeaderItem(21);
        ___qtablewidgetitem484->setText(QCoreApplication::translate("InputRZForm", "22", nullptr));
        QTableWidgetItem *___qtablewidgetitem485 = cylTable->verticalHeaderItem(22);
        ___qtablewidgetitem485->setText(QCoreApplication::translate("InputRZForm", "23", nullptr));
        QTableWidgetItem *___qtablewidgetitem486 = cylTable->verticalHeaderItem(23);
        ___qtablewidgetitem486->setText(QCoreApplication::translate("InputRZForm", "24", nullptr));
        QTableWidgetItem *___qtablewidgetitem487 = cylTable->verticalHeaderItem(24);
        ___qtablewidgetitem487->setText(QCoreApplication::translate("InputRZForm", "25", nullptr));
        QTableWidgetItem *___qtablewidgetitem488 = cylTable->verticalHeaderItem(25);
        ___qtablewidgetitem488->setText(QCoreApplication::translate("InputRZForm", "26", nullptr));
        QTableWidgetItem *___qtablewidgetitem489 = cylTable->verticalHeaderItem(26);
        ___qtablewidgetitem489->setText(QCoreApplication::translate("InputRZForm", "27", nullptr));
        QTableWidgetItem *___qtablewidgetitem490 = cylTable->verticalHeaderItem(27);
        ___qtablewidgetitem490->setText(QCoreApplication::translate("InputRZForm", "28", nullptr));
        QTableWidgetItem *___qtablewidgetitem491 = cylTable->verticalHeaderItem(28);
        ___qtablewidgetitem491->setText(QCoreApplication::translate("InputRZForm", "29", nullptr));
        QTableWidgetItem *___qtablewidgetitem492 = cylTable->verticalHeaderItem(29);
        ___qtablewidgetitem492->setText(QCoreApplication::translate("InputRZForm", "30", nullptr));
        QTableWidgetItem *___qtablewidgetitem493 = cylTable->verticalHeaderItem(30);
        ___qtablewidgetitem493->setText(QCoreApplication::translate("InputRZForm", "31", nullptr));
        QTableWidgetItem *___qtablewidgetitem494 = cylTable->verticalHeaderItem(31);
        ___qtablewidgetitem494->setText(QCoreApplication::translate("InputRZForm", "32", nullptr));
        QTableWidgetItem *___qtablewidgetitem495 = cylTable->verticalHeaderItem(32);
        ___qtablewidgetitem495->setText(QCoreApplication::translate("InputRZForm", "33", nullptr));
        QTableWidgetItem *___qtablewidgetitem496 = cylTable->verticalHeaderItem(33);
        ___qtablewidgetitem496->setText(QCoreApplication::translate("InputRZForm", "34", nullptr));
        QTableWidgetItem *___qtablewidgetitem497 = cylTable->verticalHeaderItem(34);
        ___qtablewidgetitem497->setText(QCoreApplication::translate("InputRZForm", "35", nullptr));
        QTableWidgetItem *___qtablewidgetitem498 = cylTable->verticalHeaderItem(35);
        ___qtablewidgetitem498->setText(QCoreApplication::translate("InputRZForm", "36", nullptr));
        QTableWidgetItem *___qtablewidgetitem499 = cylTable->verticalHeaderItem(36);
        ___qtablewidgetitem499->setText(QCoreApplication::translate("InputRZForm", "37", nullptr));
        QTableWidgetItem *___qtablewidgetitem500 = cylTable->verticalHeaderItem(37);
        ___qtablewidgetitem500->setText(QCoreApplication::translate("InputRZForm", "38", nullptr));
        QTableWidgetItem *___qtablewidgetitem501 = cylTable->verticalHeaderItem(38);
        ___qtablewidgetitem501->setText(QCoreApplication::translate("InputRZForm", "39", nullptr));
        QTableWidgetItem *___qtablewidgetitem502 = cylTable->verticalHeaderItem(39);
        ___qtablewidgetitem502->setText(QCoreApplication::translate("InputRZForm", "40", nullptr));
        QTableWidgetItem *___qtablewidgetitem503 = cylTable->verticalHeaderItem(40);
        ___qtablewidgetitem503->setText(QCoreApplication::translate("InputRZForm", "41", nullptr));
        QTableWidgetItem *___qtablewidgetitem504 = cylTable->verticalHeaderItem(41);
        ___qtablewidgetitem504->setText(QCoreApplication::translate("InputRZForm", "42", nullptr));
        QTableWidgetItem *___qtablewidgetitem505 = cylTable->verticalHeaderItem(42);
        ___qtablewidgetitem505->setText(QCoreApplication::translate("InputRZForm", "43", nullptr));
        QTableWidgetItem *___qtablewidgetitem506 = cylTable->verticalHeaderItem(43);
        ___qtablewidgetitem506->setText(QCoreApplication::translate("InputRZForm", "44", nullptr));
        QTableWidgetItem *___qtablewidgetitem507 = cylTable->verticalHeaderItem(44);
        ___qtablewidgetitem507->setText(QCoreApplication::translate("InputRZForm", "45", nullptr));
        QTableWidgetItem *___qtablewidgetitem508 = cylTable->verticalHeaderItem(45);
        ___qtablewidgetitem508->setText(QCoreApplication::translate("InputRZForm", "46", nullptr));
        QTableWidgetItem *___qtablewidgetitem509 = cylTable->verticalHeaderItem(46);
        ___qtablewidgetitem509->setText(QCoreApplication::translate("InputRZForm", "47", nullptr));
        QTableWidgetItem *___qtablewidgetitem510 = cylTable->verticalHeaderItem(47);
        ___qtablewidgetitem510->setText(QCoreApplication::translate("InputRZForm", "48", nullptr));
        QTableWidgetItem *___qtablewidgetitem511 = cylTable->verticalHeaderItem(48);
        ___qtablewidgetitem511->setText(QCoreApplication::translate("InputRZForm", "49", nullptr));
        QTableWidgetItem *___qtablewidgetitem512 = cylTable->verticalHeaderItem(49);
        ___qtablewidgetitem512->setText(QCoreApplication::translate("InputRZForm", "50", nullptr));
#if QT_CONFIG(tooltip)
        cylTable->setToolTip(QCoreApplication::translate("InputRZForm", "Input cylinder radii.", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        cylTable->setWhatsThis(QCoreApplication::translate("InputRZForm", "Input cylinder radii.", nullptr));
#endif // QT_CONFIG(whatsthis)
        TabWidgetRZ->setTabText(TabWidgetRZ->indexOf(Gtab), QCoreApplication::translate("InputRZForm", "Geometry", nullptr));
#if QT_CONFIG(tooltip)
        cavityGroupBox->setToolTip(QCoreApplication::translate("InputRZForm", "Define cavity geometry.", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        cavityGroupBox->setWhatsThis(QCoreApplication::translate("InputRZForm", "Define cavity geometry.", nullptr));
#endif // QT_CONFIG(whatsthis)
        cavityGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Cavity input", nullptr));
        ButtonGroup25->setTitle(QCoreApplication::translate("InputRZForm", "chamber type", nullptr));
        thimbleRadioButton->setText(QCoreApplication::translate("InputRZForm", "thimble", nullptr));
        parallelplateRadioButton->setText(QCoreApplication::translate("InputRZForm", "parallel plate", nullptr));
        wallLabel->setText(QCoreApplication::translate("InputRZForm", "Wall thickness", nullptr));
        wallthickEdit->setText(QCoreApplication::translate("InputRZForm", "0.1", nullptr));
        cavradLabel->setText(QCoreApplication::translate("InputRZForm", "Cavity radius", nullptr));
        cavradEdit->setText(QCoreApplication::translate("InputRZForm", "1.0", nullptr));
        cavlenLabel->setText(QCoreApplication::translate("InputRZForm", "Cavity length", nullptr));
        cavlenEdit->setText(QCoreApplication::translate("InputRZForm", "2.0", nullptr));
        electradLabel->setText(QCoreApplication::translate("InputRZForm", "Electrode radius", nullptr));
        electradEdit->setText(QCoreApplication::translate("InputRZForm", "0.01", nullptr));
        wallmaterialLabel->setText(QCoreApplication::translate("InputRZForm", "Wall material", nullptr));
        electrmatLabel->setText(QCoreApplication::translate("InputRZForm", "Electrode material", nullptr));
#if QT_CONFIG(tooltip)
        gr_indGroupBox->setToolTip(QString());
#endif // QT_CONFIG(tooltip)
        gr_indGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Group or individual input", nullptr));
        GroupBox56->setTitle(QCoreApplication::translate("InputRZForm", "Regions comprising cavity", nullptr));
        QTableWidgetItem *___qtablewidgetitem513 = cavTable->horizontalHeaderItem(0);
        ___qtablewidgetitem513->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem514 = cavTable->verticalHeaderItem(0);
        ___qtablewidgetitem514->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem515 = cavTable->verticalHeaderItem(1);
        ___qtablewidgetitem515->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem516 = cavTable->verticalHeaderItem(2);
        ___qtablewidgetitem516->setText(QCoreApplication::translate("InputRZForm", "3", nullptr));
        QTableWidgetItem *___qtablewidgetitem517 = cavTable->verticalHeaderItem(3);
        ___qtablewidgetitem517->setText(QCoreApplication::translate("InputRZForm", "4", nullptr));
        QTableWidgetItem *___qtablewidgetitem518 = cavTable->verticalHeaderItem(4);
        ___qtablewidgetitem518->setText(QCoreApplication::translate("InputRZForm", "5", nullptr));
        QTableWidgetItem *___qtablewidgetitem519 = cavTable->verticalHeaderItem(5);
        ___qtablewidgetitem519->setText(QCoreApplication::translate("InputRZForm", "6", nullptr));
        QTableWidgetItem *___qtablewidgetitem520 = cavTable->verticalHeaderItem(6);
        ___qtablewidgetitem520->setText(QCoreApplication::translate("InputRZForm", "7", nullptr));
        QTableWidgetItem *___qtablewidgetitem521 = cavTable->verticalHeaderItem(7);
        ___qtablewidgetitem521->setText(QCoreApplication::translate("InputRZForm", "8", nullptr));
        QTableWidgetItem *___qtablewidgetitem522 = cavTable->verticalHeaderItem(8);
        ___qtablewidgetitem522->setText(QCoreApplication::translate("InputRZForm", "9", nullptr));
        QTableWidgetItem *___qtablewidgetitem523 = cavTable->verticalHeaderItem(9);
        ___qtablewidgetitem523->setText(QCoreApplication::translate("InputRZForm", "10", nullptr));
        QTableWidgetItem *___qtablewidgetitem524 = cavTable->verticalHeaderItem(10);
        ___qtablewidgetitem524->setText(QCoreApplication::translate("InputRZForm", "11", nullptr));
        QTableWidgetItem *___qtablewidgetitem525 = cavTable->verticalHeaderItem(11);
        ___qtablewidgetitem525->setText(QCoreApplication::translate("InputRZForm", "12", nullptr));
        QTableWidgetItem *___qtablewidgetitem526 = cavTable->verticalHeaderItem(12);
        ___qtablewidgetitem526->setText(QCoreApplication::translate("InputRZForm", "13", nullptr));
        QTableWidgetItem *___qtablewidgetitem527 = cavTable->verticalHeaderItem(13);
        ___qtablewidgetitem527->setText(QCoreApplication::translate("InputRZForm", "14", nullptr));
        QTableWidgetItem *___qtablewidgetitem528 = cavTable->verticalHeaderItem(14);
        ___qtablewidgetitem528->setText(QCoreApplication::translate("InputRZForm", "15", nullptr));
        QTableWidgetItem *___qtablewidgetitem529 = cavTable->verticalHeaderItem(15);
        ___qtablewidgetitem529->setText(QCoreApplication::translate("InputRZForm", "16", nullptr));
        QTableWidgetItem *___qtablewidgetitem530 = cavTable->verticalHeaderItem(16);
        ___qtablewidgetitem530->setText(QCoreApplication::translate("InputRZForm", "17", nullptr));
        QTableWidgetItem *___qtablewidgetitem531 = cavTable->verticalHeaderItem(17);
        ___qtablewidgetitem531->setText(QCoreApplication::translate("InputRZForm", "18", nullptr));
        QTableWidgetItem *___qtablewidgetitem532 = cavTable->verticalHeaderItem(18);
        ___qtablewidgetitem532->setText(QCoreApplication::translate("InputRZForm", "19", nullptr));
        QTableWidgetItem *___qtablewidgetitem533 = cavTable->verticalHeaderItem(19);
        ___qtablewidgetitem533->setText(QCoreApplication::translate("InputRZForm", "20", nullptr));
        QTableWidgetItem *___qtablewidgetitem534 = cavTable->verticalHeaderItem(20);
        ___qtablewidgetitem534->setText(QCoreApplication::translate("InputRZForm", "21", nullptr));
        QTableWidgetItem *___qtablewidgetitem535 = cavTable->verticalHeaderItem(21);
        ___qtablewidgetitem535->setText(QCoreApplication::translate("InputRZForm", "22", nullptr));
        QTableWidgetItem *___qtablewidgetitem536 = cavTable->verticalHeaderItem(22);
        ___qtablewidgetitem536->setText(QCoreApplication::translate("InputRZForm", "23", nullptr));
        QTableWidgetItem *___qtablewidgetitem537 = cavTable->verticalHeaderItem(23);
        ___qtablewidgetitem537->setText(QCoreApplication::translate("InputRZForm", "24", nullptr));
        QTableWidgetItem *___qtablewidgetitem538 = cavTable->verticalHeaderItem(24);
        ___qtablewidgetitem538->setText(QCoreApplication::translate("InputRZForm", "25", nullptr));
        QTableWidgetItem *___qtablewidgetitem539 = cavTable->verticalHeaderItem(25);
        ___qtablewidgetitem539->setText(QCoreApplication::translate("InputRZForm", "26", nullptr));
        QTableWidgetItem *___qtablewidgetitem540 = cavTable->verticalHeaderItem(26);
        ___qtablewidgetitem540->setText(QCoreApplication::translate("InputRZForm", "27", nullptr));
        QTableWidgetItem *___qtablewidgetitem541 = cavTable->verticalHeaderItem(27);
        ___qtablewidgetitem541->setText(QCoreApplication::translate("InputRZForm", "28", nullptr));
        QTableWidgetItem *___qtablewidgetitem542 = cavTable->verticalHeaderItem(28);
        ___qtablewidgetitem542->setText(QCoreApplication::translate("InputRZForm", "29", nullptr));
        QTableWidgetItem *___qtablewidgetitem543 = cavTable->verticalHeaderItem(29);
        ___qtablewidgetitem543->setText(QCoreApplication::translate("InputRZForm", "30", nullptr));
        QTableWidgetItem *___qtablewidgetitem544 = cavTable->verticalHeaderItem(30);
        ___qtablewidgetitem544->setText(QCoreApplication::translate("InputRZForm", "31", nullptr));
        QTableWidgetItem *___qtablewidgetitem545 = cavTable->verticalHeaderItem(31);
        ___qtablewidgetitem545->setText(QCoreApplication::translate("InputRZForm", "32", nullptr));
        QTableWidgetItem *___qtablewidgetitem546 = cavTable->verticalHeaderItem(32);
        ___qtablewidgetitem546->setText(QCoreApplication::translate("InputRZForm", "33", nullptr));
        QTableWidgetItem *___qtablewidgetitem547 = cavTable->verticalHeaderItem(33);
        ___qtablewidgetitem547->setText(QCoreApplication::translate("InputRZForm", "34", nullptr));
        QTableWidgetItem *___qtablewidgetitem548 = cavTable->verticalHeaderItem(34);
        ___qtablewidgetitem548->setText(QCoreApplication::translate("InputRZForm", "35", nullptr));
        QTableWidgetItem *___qtablewidgetitem549 = cavTable->verticalHeaderItem(35);
        ___qtablewidgetitem549->setText(QCoreApplication::translate("InputRZForm", "36", nullptr));
        QTableWidgetItem *___qtablewidgetitem550 = cavTable->verticalHeaderItem(36);
        ___qtablewidgetitem550->setText(QCoreApplication::translate("InputRZForm", "37", nullptr));
        QTableWidgetItem *___qtablewidgetitem551 = cavTable->verticalHeaderItem(37);
        ___qtablewidgetitem551->setText(QCoreApplication::translate("InputRZForm", "38", nullptr));
        QTableWidgetItem *___qtablewidgetitem552 = cavTable->verticalHeaderItem(38);
        ___qtablewidgetitem552->setText(QCoreApplication::translate("InputRZForm", "39", nullptr));
        QTableWidgetItem *___qtablewidgetitem553 = cavTable->verticalHeaderItem(39);
        ___qtablewidgetitem553->setText(QCoreApplication::translate("InputRZForm", "40", nullptr));
        QTableWidgetItem *___qtablewidgetitem554 = cavTable->verticalHeaderItem(40);
        ___qtablewidgetitem554->setText(QCoreApplication::translate("InputRZForm", "41", nullptr));
        QTableWidgetItem *___qtablewidgetitem555 = cavTable->verticalHeaderItem(41);
        ___qtablewidgetitem555->setText(QCoreApplication::translate("InputRZForm", "42", nullptr));
        QTableWidgetItem *___qtablewidgetitem556 = cavTable->verticalHeaderItem(42);
        ___qtablewidgetitem556->setText(QCoreApplication::translate("InputRZForm", "43", nullptr));
        QTableWidgetItem *___qtablewidgetitem557 = cavTable->verticalHeaderItem(43);
        ___qtablewidgetitem557->setText(QCoreApplication::translate("InputRZForm", "44", nullptr));
        QTableWidgetItem *___qtablewidgetitem558 = cavTable->verticalHeaderItem(44);
        ___qtablewidgetitem558->setText(QCoreApplication::translate("InputRZForm", "45", nullptr));
        QTableWidgetItem *___qtablewidgetitem559 = cavTable->verticalHeaderItem(45);
        ___qtablewidgetitem559->setText(QCoreApplication::translate("InputRZForm", "46", nullptr));
        QTableWidgetItem *___qtablewidgetitem560 = cavTable->verticalHeaderItem(46);
        ___qtablewidgetitem560->setText(QCoreApplication::translate("InputRZForm", "47", nullptr));
        QTableWidgetItem *___qtablewidgetitem561 = cavTable->verticalHeaderItem(47);
        ___qtablewidgetitem561->setText(QCoreApplication::translate("InputRZForm", "48", nullptr));
        QTableWidgetItem *___qtablewidgetitem562 = cavTable->verticalHeaderItem(48);
        ___qtablewidgetitem562->setText(QCoreApplication::translate("InputRZForm", "49", nullptr));
        QTableWidgetItem *___qtablewidgetitem563 = cavTable->verticalHeaderItem(49);
        ___qtablewidgetitem563->setText(QCoreApplication::translate("InputRZForm", "50", nullptr));
        TabWidgetRZ->setTabText(TabWidgetRZ->indexOf(CItab), QCoreApplication::translate("InputRZForm", "Cavity", nullptr));
#if QT_CONFIG(tooltip)
        ELGroupBox->setToolTip(QCoreApplication::translate("InputRZForm", "Specify upper and lower energy limits for all PEGSless data", nullptr));
#endif // QT_CONFIG(tooltip)
        ELGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "energy limits (MeV)", nullptr));
        AELabel->setText(QCoreApplication::translate("InputRZForm", "AE", nullptr));
#if QT_CONFIG(tooltip)
        AEEdit->setToolTip(QCoreApplication::translate("InputRZForm", "minimum total e- energy for cross-sections", nullptr));
#endif // QT_CONFIG(tooltip)
        APLabel->setText(QCoreApplication::translate("InputRZForm", "AP", nullptr));
#if QT_CONFIG(tooltip)
        APEdit->setToolTip(QCoreApplication::translate("InputRZForm", "minimum photon energy for cross-sections", nullptr));
#endif // QT_CONFIG(tooltip)
        UELabel->setText(QCoreApplication::translate("InputRZForm", "UE", nullptr));
#if QT_CONFIG(tooltip)
        UEEdit->setToolTip(QCoreApplication::translate("InputRZForm", "maximum e- kinetic energy for cross-sections", nullptr));
#endif // QT_CONFIG(tooltip)
        UPLabel->setText(QCoreApplication::translate("InputRZForm", "UP", nullptr));
#if QT_CONFIG(tooltip)
        UPEdit->setToolTip(QCoreApplication::translate("InputRZForm", "maximum photon energy for cross-sections", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        MDGroupBox->setToolTip(QCoreApplication::translate("InputRZForm", "<html><head/><body><p>Use this if PEGSless media specified in a material data file. Additional media can be specified in the .egsinp file (see below).</p></body></html>", nullptr));
#endif // QT_CONFIG(tooltip)
        MDGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "material data file name", nullptr));
#if QT_CONFIG(tooltip)
        MDFileButton->setToolTip(QCoreApplication::translate("InputRZForm", "Browse to select a material data file.\n"
"Browsing starts in your $EGS_HOME/pegs4/data directory.", nullptr));
#endif // QT_CONFIG(tooltip)
        MDFileButton->setText(QCoreApplication::translate("InputRZForm", "...", nullptr));
        inpmediaGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "media defined in .egsinp file", nullptr));
#if QT_CONFIG(tooltip)
        compositionGroupBox->setToolTip(QCoreApplication::translate("InputRZForm", "specify medium composition", nullptr));
#endif // QT_CONFIG(tooltip)
        compositionGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Medium composition", nullptr));
        QTableWidgetItem *___qtablewidgetitem564 = pz_or_rhozTable->horizontalHeaderItem(0);
        ___qtablewidgetitem564->setText(QCoreApplication::translate("InputRZForm", "Element", nullptr));
        QTableWidgetItem *___qtablewidgetitem565 = pz_or_rhozTable->horizontalHeaderItem(1);
        ___qtablewidgetitem565->setText(QCoreApplication::translate("InputRZForm", "Fraction by weight", nullptr));
        QTableWidgetItem *___qtablewidgetitem566 = pz_or_rhozTable->verticalHeaderItem(0);
        ___qtablewidgetitem566->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem567 = pz_or_rhozTable->verticalHeaderItem(1);
        ___qtablewidgetitem567->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem568 = pz_or_rhozTable->verticalHeaderItem(2);
        ___qtablewidgetitem568->setText(QCoreApplication::translate("InputRZForm", "3", nullptr));
        QTableWidgetItem *___qtablewidgetitem569 = pz_or_rhozTable->verticalHeaderItem(3);
        ___qtablewidgetitem569->setText(QCoreApplication::translate("InputRZForm", "4", nullptr));
        QTableWidgetItem *___qtablewidgetitem570 = pz_or_rhozTable->verticalHeaderItem(4);
        ___qtablewidgetitem570->setText(QCoreApplication::translate("InputRZForm", "5", nullptr));
        QTableWidgetItem *___qtablewidgetitem571 = pz_or_rhozTable->verticalHeaderItem(5);
        ___qtablewidgetitem571->setText(QCoreApplication::translate("InputRZForm", "6", nullptr));
        QTableWidgetItem *___qtablewidgetitem572 = pz_or_rhozTable->verticalHeaderItem(6);
        ___qtablewidgetitem572->setText(QCoreApplication::translate("InputRZForm", "7", nullptr));
        QTableWidgetItem *___qtablewidgetitem573 = pz_or_rhozTable->verticalHeaderItem(7);
        ___qtablewidgetitem573->setText(QCoreApplication::translate("InputRZForm", "8", nullptr));
        QTableWidgetItem *___qtablewidgetitem574 = pz_or_rhozTable->verticalHeaderItem(8);
        ___qtablewidgetitem574->setText(QCoreApplication::translate("InputRZForm", "9", nullptr));
        QTableWidgetItem *___qtablewidgetitem575 = pz_or_rhozTable->verticalHeaderItem(9);
        ___qtablewidgetitem575->setText(QCoreApplication::translate("InputRZForm", "10", nullptr));
        QTableWidgetItem *___qtablewidgetitem576 = pz_or_rhozTable->verticalHeaderItem(10);
        ___qtablewidgetitem576->setText(QCoreApplication::translate("InputRZForm", "11", nullptr));
        QTableWidgetItem *___qtablewidgetitem577 = pz_or_rhozTable->verticalHeaderItem(11);
        ___qtablewidgetitem577->setText(QCoreApplication::translate("InputRZForm", "12", nullptr));
        QTableWidgetItem *___qtablewidgetitem578 = pz_or_rhozTable->verticalHeaderItem(12);
        ___qtablewidgetitem578->setText(QCoreApplication::translate("InputRZForm", "13", nullptr));
        QTableWidgetItem *___qtablewidgetitem579 = pz_or_rhozTable->verticalHeaderItem(13);
        ___qtablewidgetitem579->setText(QCoreApplication::translate("InputRZForm", "14", nullptr));
        QTableWidgetItem *___qtablewidgetitem580 = pz_or_rhozTable->verticalHeaderItem(14);
        ___qtablewidgetitem580->setText(QCoreApplication::translate("InputRZForm", "15", nullptr));
        QTableWidgetItem *___qtablewidgetitem581 = pz_or_rhozTable->verticalHeaderItem(15);
        ___qtablewidgetitem581->setText(QCoreApplication::translate("InputRZForm", "16", nullptr));
        QTableWidgetItem *___qtablewidgetitem582 = pz_or_rhozTable->verticalHeaderItem(16);
        ___qtablewidgetitem582->setText(QCoreApplication::translate("InputRZForm", "17", nullptr));
        QTableWidgetItem *___qtablewidgetitem583 = pz_or_rhozTable->verticalHeaderItem(17);
        ___qtablewidgetitem583->setText(QCoreApplication::translate("InputRZForm", "18", nullptr));
        QTableWidgetItem *___qtablewidgetitem584 = pz_or_rhozTable->verticalHeaderItem(18);
        ___qtablewidgetitem584->setText(QCoreApplication::translate("InputRZForm", "19", nullptr));
        QTableWidgetItem *___qtablewidgetitem585 = pz_or_rhozTable->verticalHeaderItem(19);
        ___qtablewidgetitem585->setText(QCoreApplication::translate("InputRZForm", "20", nullptr));
        QTableWidgetItem *___qtablewidgetitem586 = pz_or_rhozTable->verticalHeaderItem(20);
        ___qtablewidgetitem586->setText(QCoreApplication::translate("InputRZForm", "21", nullptr));
        QTableWidgetItem *___qtablewidgetitem587 = pz_or_rhozTable->verticalHeaderItem(21);
        ___qtablewidgetitem587->setText(QCoreApplication::translate("InputRZForm", "22", nullptr));
        QTableWidgetItem *___qtablewidgetitem588 = pz_or_rhozTable->verticalHeaderItem(22);
        ___qtablewidgetitem588->setText(QCoreApplication::translate("InputRZForm", "23", nullptr));
        QTableWidgetItem *___qtablewidgetitem589 = pz_or_rhozTable->verticalHeaderItem(23);
        ___qtablewidgetitem589->setText(QCoreApplication::translate("InputRZForm", "24", nullptr));
        QTableWidgetItem *___qtablewidgetitem590 = pz_or_rhozTable->verticalHeaderItem(24);
        ___qtablewidgetitem590->setText(QCoreApplication::translate("InputRZForm", "25", nullptr));
        QTableWidgetItem *___qtablewidgetitem591 = pz_or_rhozTable->verticalHeaderItem(25);
        ___qtablewidgetitem591->setText(QCoreApplication::translate("InputRZForm", "26", nullptr));
        QTableWidgetItem *___qtablewidgetitem592 = pz_or_rhozTable->verticalHeaderItem(26);
        ___qtablewidgetitem592->setText(QCoreApplication::translate("InputRZForm", "27", nullptr));
        QTableWidgetItem *___qtablewidgetitem593 = pz_or_rhozTable->verticalHeaderItem(27);
        ___qtablewidgetitem593->setText(QCoreApplication::translate("InputRZForm", "28", nullptr));
        QTableWidgetItem *___qtablewidgetitem594 = pz_or_rhozTable->verticalHeaderItem(28);
        ___qtablewidgetitem594->setText(QCoreApplication::translate("InputRZForm", "29", nullptr));
        QTableWidgetItem *___qtablewidgetitem595 = pz_or_rhozTable->verticalHeaderItem(29);
        ___qtablewidgetitem595->setText(QCoreApplication::translate("InputRZForm", "30", nullptr));
        QTableWidgetItem *___qtablewidgetitem596 = pz_or_rhozTable->verticalHeaderItem(30);
        ___qtablewidgetitem596->setText(QCoreApplication::translate("InputRZForm", "31", nullptr));
        QTableWidgetItem *___qtablewidgetitem597 = pz_or_rhozTable->verticalHeaderItem(31);
        ___qtablewidgetitem597->setText(QCoreApplication::translate("InputRZForm", "32", nullptr));
        QTableWidgetItem *___qtablewidgetitem598 = pz_or_rhozTable->verticalHeaderItem(32);
        ___qtablewidgetitem598->setText(QCoreApplication::translate("InputRZForm", "33", nullptr));
        QTableWidgetItem *___qtablewidgetitem599 = pz_or_rhozTable->verticalHeaderItem(33);
        ___qtablewidgetitem599->setText(QCoreApplication::translate("InputRZForm", "34", nullptr));
        QTableWidgetItem *___qtablewidgetitem600 = pz_or_rhozTable->verticalHeaderItem(34);
        ___qtablewidgetitem600->setText(QCoreApplication::translate("InputRZForm", "35", nullptr));
        QTableWidgetItem *___qtablewidgetitem601 = pz_or_rhozTable->verticalHeaderItem(35);
        ___qtablewidgetitem601->setText(QCoreApplication::translate("InputRZForm", "36", nullptr));
        QTableWidgetItem *___qtablewidgetitem602 = pz_or_rhozTable->verticalHeaderItem(36);
        ___qtablewidgetitem602->setText(QCoreApplication::translate("InputRZForm", "37", nullptr));
        QTableWidgetItem *___qtablewidgetitem603 = pz_or_rhozTable->verticalHeaderItem(37);
        ___qtablewidgetitem603->setText(QCoreApplication::translate("InputRZForm", "38", nullptr));
        QTableWidgetItem *___qtablewidgetitem604 = pz_or_rhozTable->verticalHeaderItem(38);
        ___qtablewidgetitem604->setText(QCoreApplication::translate("InputRZForm", "39", nullptr));
        QTableWidgetItem *___qtablewidgetitem605 = pz_or_rhozTable->verticalHeaderItem(39);
        ___qtablewidgetitem605->setText(QCoreApplication::translate("InputRZForm", "40", nullptr));
        QTableWidgetItem *___qtablewidgetitem606 = pz_or_rhozTable->verticalHeaderItem(40);
        ___qtablewidgetitem606->setText(QCoreApplication::translate("InputRZForm", "41", nullptr));
        QTableWidgetItem *___qtablewidgetitem607 = pz_or_rhozTable->verticalHeaderItem(41);
        ___qtablewidgetitem607->setText(QCoreApplication::translate("InputRZForm", "42", nullptr));
        QTableWidgetItem *___qtablewidgetitem608 = pz_or_rhozTable->verticalHeaderItem(42);
        ___qtablewidgetitem608->setText(QCoreApplication::translate("InputRZForm", "43", nullptr));
        QTableWidgetItem *___qtablewidgetitem609 = pz_or_rhozTable->verticalHeaderItem(43);
        ___qtablewidgetitem609->setText(QCoreApplication::translate("InputRZForm", "44", nullptr));
        QTableWidgetItem *___qtablewidgetitem610 = pz_or_rhozTable->verticalHeaderItem(44);
        ___qtablewidgetitem610->setText(QCoreApplication::translate("InputRZForm", "45", nullptr));
        QTableWidgetItem *___qtablewidgetitem611 = pz_or_rhozTable->verticalHeaderItem(45);
        ___qtablewidgetitem611->setText(QCoreApplication::translate("InputRZForm", "46", nullptr));
        QTableWidgetItem *___qtablewidgetitem612 = pz_or_rhozTable->verticalHeaderItem(46);
        ___qtablewidgetitem612->setText(QCoreApplication::translate("InputRZForm", "47", nullptr));
        QTableWidgetItem *___qtablewidgetitem613 = pz_or_rhozTable->verticalHeaderItem(47);
        ___qtablewidgetitem613->setText(QCoreApplication::translate("InputRZForm", "48", nullptr));
        QTableWidgetItem *___qtablewidgetitem614 = pz_or_rhozTable->verticalHeaderItem(48);
        ___qtablewidgetitem614->setText(QCoreApplication::translate("InputRZForm", "49", nullptr));
        QTableWidgetItem *___qtablewidgetitem615 = pz_or_rhozTable->verticalHeaderItem(49);
        ___qtablewidgetitem615->setText(QCoreApplication::translate("InputRZForm", "50", nullptr));
#if QT_CONFIG(tooltip)
        pz_or_rhozTable->setToolTip(QCoreApplication::translate("InputRZForm", "Elements must be specified by their chemical symbol all uppercase.\n"
"For mixtures, composition is specified by fraction by weight of each element.\n"
"For compounds, composition is specified by stoichiometric index of each element.", nullptr));
#endif // QT_CONFIG(tooltip)
        medNameGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Medium name", nullptr));
#if QT_CONFIG(tooltip)
        inpmediumComboBox->setToolTip(QString());
#endif // QT_CONFIG(tooltip)
        medTypeGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Medium type", nullptr));
        medTypeComboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "Element", nullptr));
        medTypeComboBox->setItemText(1, QCoreApplication::translate("InputRZForm", "Compound", nullptr));
        medTypeComboBox->setItemText(2, QCoreApplication::translate("InputRZForm", "Mixture", nullptr));

#if QT_CONFIG(tooltip)
        medTypeComboBox->setToolTip(QCoreApplication::translate("InputRZForm", "Select 'Mixture' to specify composition by fraction by weight,\n"
"'Compound' to specify by stoichiometric index", nullptr));
#endif // QT_CONFIG(tooltip)
        rhoGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Mass density", nullptr));
#if QT_CONFIG(tooltip)
        rhoEdit->setToolTip(QCoreApplication::translate("InputRZForm", "medium density in g/cm^3 or kg/cm^3", nullptr));
#endif // QT_CONFIG(tooltip)
        rhoScaleComboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "g/cm^3", nullptr));
        rhoScaleComboBox->setItemText(1, QCoreApplication::translate("InputRZForm", "kg/cm^3", nullptr));

        medOptionsGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Options", nullptr));
#if QT_CONFIG(tooltip)
        DCcheckBox->setToolTip(QCoreApplication::translate("InputRZForm", "Select to specify an ICRU density correction file (see below).", nullptr));
#endif // QT_CONFIG(tooltip)
        DCcheckBox->setText(QCoreApplication::translate("InputRZForm", "ICRU density correction", nullptr));
#if QT_CONFIG(tooltip)
        ICRUradCheckBox->setToolTip(QCoreApplication::translate("InputRZForm", "Select to use ICRU corrections to brems cross-sections.\n"
"Otherwise, Koch-Motz (KM) corrections will be used. ", nullptr));
#endif // QT_CONFIG(tooltip)
        ICRUradCheckBox->setText(QCoreApplication::translate("InputRZForm", "ICRU radiative stopping power", nullptr));
#if QT_CONFIG(tooltip)
        isGasCheckBox->setToolTip(QCoreApplication::translate("InputRZForm", "Check this if the medium is a gas.\n"
"This will enable you to input the gas pressure of the medium.", nullptr));
#endif // QT_CONFIG(tooltip)
        isGasCheckBox->setText(QCoreApplication::translate("InputRZForm", "medium is a gas", nullptr));
        gaspLabel->setText(QCoreApplication::translate("InputRZForm", "gas pressure", nullptr));
#if QT_CONFIG(tooltip)
        gaspEdit->setToolTip(QCoreApplication::translate("InputRZForm", "If the medium is a gas, specify gas pressure (atm) here.\n"
"Defaults to 1 atm. if left blank or zero.", nullptr));
#endif // QT_CONFIG(tooltip)
        gaspUnits->setText(QCoreApplication::translate("InputRZForm", "atm", nullptr));
#if QT_CONFIG(tooltip)
        DFgroupBox->setToolTip(QCoreApplication::translate("InputRZForm", "Specify a density correction file which, when applied to calculated cross-sections,\n"
"results in agreement with stopping powers published in ICRU37.  Note that medium\n"
"composition and density as specified in the density correction file are used.", nullptr));
#endif // QT_CONFIG(tooltip)
        DFgroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Density correction file", nullptr));
        DFSearchLabel->setText(QCoreApplication::translate("InputRZForm", "Look in", nullptr));
        DFSearchComboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "EGS_HOME", nullptr));
        DFSearchComboBox->setItemText(1, QCoreApplication::translate("InputRZForm", "HEN_HOUSE", nullptr));

#if QT_CONFIG(tooltip)
        DFSearchComboBox->setToolTip(QCoreApplication::translate("InputRZForm", "Select in which directory to begin browsing for the density correction file.\n"
"Will start in /pegs4/density_corrections/ subdirectory off the directory selected here.", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        DFEdit->setToolTip(QCoreApplication::translate("InputRZForm", "Full name of density correction file.", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        DFBrowse->setToolTip(QCoreApplication::translate("InputRZForm", "Browse for density correction file.", nullptr));
#endif // QT_CONFIG(tooltip)
        DFBrowse->setText(QCoreApplication::translate("InputRZForm", "...", nullptr));
        TabWidgetRZ->setTabText(TabWidgetRZ->indexOf(MDTab), QCoreApplication::translate("InputRZForm", "Media Definition", nullptr));
        in_particleButtonGroup->setTitle(QCoreApplication::translate("InputRZForm", "Incident particle", nullptr));
        eRadioButton->setText(QCoreApplication::translate("InputRZForm", "electron", nullptr));
        phRadioButton->setText(QCoreApplication::translate("InputRZForm", "photon", nullptr));
        pRadioButton->setText(QCoreApplication::translate("InputRZForm", "positron", nullptr));
#if QT_CONFIG(tooltip)
        chargedRadioButton->setToolTip(QCoreApplication::translate("InputRZForm", "available only for source 21 and 22", nullptr));
#endif // QT_CONFIG(tooltip)
        chargedRadioButton->setText(QCoreApplication::translate("InputRZForm", "charged", nullptr));
#if QT_CONFIG(tooltip)
        allRadioButton->setToolTip(QCoreApplication::translate("InputRZForm", "available only for source 21 and 22", nullptr));
#endif // QT_CONFIG(tooltip)
        allRadioButton->setText(QCoreApplication::translate("InputRZForm", "all", nullptr));
        energyButtonGroup->setTitle(QCoreApplication::translate("InputRZForm", "Incident energy", nullptr));
        monoenergeticRadioButton->setText(QCoreApplication::translate("InputRZForm", "monoenergetic", nullptr));
        spectrumRadioButton->setText(QCoreApplication::translate("InputRZForm", "spectrum", nullptr));
        IniEgroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Incident kinetic E [MeV]", nullptr));
        specfnameGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Spectrum file name", nullptr));
        specfnameComboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "co60.spectrum", nullptr));

        specfnameButton->setText(QCoreApplication::translate("InputRZForm", "...", nullptr));
        ioutspGroupBox->setTitle(QString());
        ioutspCheckBox->setText(QCoreApplication::translate("InputRZForm", "Include spectrum in .egslst", nullptr));
        sourceLabel->setText(QCoreApplication::translate("InputRZForm", "source number", nullptr));
        sourceComboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "0", nullptr));
        sourceComboBox->setItemText(1, QCoreApplication::translate("InputRZForm", "1", nullptr));
        sourceComboBox->setItemText(2, QCoreApplication::translate("InputRZForm", "2", nullptr));
        sourceComboBox->setItemText(3, QCoreApplication::translate("InputRZForm", "3", nullptr));
        sourceComboBox->setItemText(4, QCoreApplication::translate("InputRZForm", "4", nullptr));
        sourceComboBox->setItemText(5, QCoreApplication::translate("InputRZForm", "10", nullptr));
        sourceComboBox->setItemText(6, QCoreApplication::translate("InputRZForm", "11", nullptr));
        sourceComboBox->setItemText(7, QCoreApplication::translate("InputRZForm", "12", nullptr));
        sourceComboBox->setItemText(8, QCoreApplication::translate("InputRZForm", "13", nullptr));
        sourceComboBox->setItemText(9, QCoreApplication::translate("InputRZForm", "14", nullptr));
        sourceComboBox->setItemText(10, QCoreApplication::translate("InputRZForm", "15", nullptr));
        sourceComboBox->setItemText(11, QCoreApplication::translate("InputRZForm", "16", nullptr));
        sourceComboBox->setItemText(12, QCoreApplication::translate("InputRZForm", "20", nullptr));
        sourceComboBox->setItemText(13, QCoreApplication::translate("InputRZForm", "21", nullptr));
        sourceComboBox->setItemText(14, QCoreApplication::translate("InputRZForm", "22", nullptr));
        sourceComboBox->setItemText(15, QCoreApplication::translate("InputRZForm", "23", nullptr));

        sourceoptionsGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Source options", nullptr));
        temp1Label->setText(QCoreApplication::translate("InputRZForm", "temp1", nullptr));
#if QT_CONFIG(tooltip)
        temp1Edit->setToolTip(QCoreApplication::translate("InputRZForm", "change depending on source", nullptr));
#endif // QT_CONFIG(tooltip)
        temp2Label->setText(QCoreApplication::translate("InputRZForm", "temp2", nullptr));
#if QT_CONFIG(tooltip)
        temp2Edit->setToolTip(QCoreApplication::translate("InputRZForm", "change depending on source", nullptr));
#endif // QT_CONFIG(tooltip)
        temp3Label->setText(QCoreApplication::translate("InputRZForm", "temp3", nullptr));
#if QT_CONFIG(tooltip)
        temp3Edit->setToolTip(QCoreApplication::translate("InputRZForm", "change depending on source", nullptr));
#endif // QT_CONFIG(tooltip)
        temp4Label->setText(QCoreApplication::translate("InputRZForm", "temp4", nullptr));
#if QT_CONFIG(tooltip)
        temp4Edit->setToolTip(QCoreApplication::translate("InputRZForm", "change depending on source", nullptr));
#endif // QT_CONFIG(tooltip)
        temp5Label->setText(QCoreApplication::translate("InputRZForm", "temp5", nullptr));
        temp6Label->setText(QCoreApplication::translate("InputRZForm", "temp6", nullptr));
        temp7Label->setText(QCoreApplication::translate("InputRZForm", "temp7", nullptr));
        phasespaceGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Phase space file", nullptr));
        phasespaceComboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "dummy.file", nullptr));

        phasespacePushButton->setText(QCoreApplication::translate("InputRZForm", "...", nullptr));
        src20GroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Radial Distribution Input", nullptr));
        srcinpmodeButtonGroup->setTitle(QCoreApplication::translate("InputRZForm", "Input mode", nullptr));
        localRadioButton->setText(QCoreApplication::translate("InputRZForm", "Local", nullptr));
        externalRadioButton->setText(QCoreApplication::translate("InputRZForm", "External", nullptr));
        QTableWidgetItem *___qtablewidgetitem616 = raddistTable->horizontalHeaderItem(0);
        ___qtablewidgetitem616->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem617 = raddistTable->horizontalHeaderItem(1);
        ___qtablewidgetitem617->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem618 = raddistTable->verticalHeaderItem(0);
        ___qtablewidgetitem618->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem619 = raddistTable->verticalHeaderItem(1);
        ___qtablewidgetitem619->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem620 = raddistTable->verticalHeaderItem(2);
        ___qtablewidgetitem620->setText(QCoreApplication::translate("InputRZForm", "3", nullptr));
        QTableWidgetItem *___qtablewidgetitem621 = raddistTable->verticalHeaderItem(3);
        ___qtablewidgetitem621->setText(QCoreApplication::translate("InputRZForm", "4", nullptr));
        QTableWidgetItem *___qtablewidgetitem622 = raddistTable->verticalHeaderItem(4);
        ___qtablewidgetitem622->setText(QCoreApplication::translate("InputRZForm", "5", nullptr));
        QTableWidgetItem *___qtablewidgetitem623 = raddistTable->verticalHeaderItem(5);
        ___qtablewidgetitem623->setText(QCoreApplication::translate("InputRZForm", "6", nullptr));
        QTableWidgetItem *___qtablewidgetitem624 = raddistTable->verticalHeaderItem(6);
        ___qtablewidgetitem624->setText(QCoreApplication::translate("InputRZForm", "7", nullptr));
        QTableWidgetItem *___qtablewidgetitem625 = raddistTable->verticalHeaderItem(7);
        ___qtablewidgetitem625->setText(QCoreApplication::translate("InputRZForm", "8", nullptr));
        QTableWidgetItem *___qtablewidgetitem626 = raddistTable->verticalHeaderItem(8);
        ___qtablewidgetitem626->setText(QCoreApplication::translate("InputRZForm", "9", nullptr));
        QTableWidgetItem *___qtablewidgetitem627 = raddistTable->verticalHeaderItem(9);
        ___qtablewidgetitem627->setText(QCoreApplication::translate("InputRZForm", "10", nullptr));
        QTableWidgetItem *___qtablewidgetitem628 = raddistTable->verticalHeaderItem(10);
        ___qtablewidgetitem628->setText(QCoreApplication::translate("InputRZForm", "11", nullptr));
        QTableWidgetItem *___qtablewidgetitem629 = raddistTable->verticalHeaderItem(11);
        ___qtablewidgetitem629->setText(QCoreApplication::translate("InputRZForm", "12", nullptr));
        QTableWidgetItem *___qtablewidgetitem630 = raddistTable->verticalHeaderItem(12);
        ___qtablewidgetitem630->setText(QCoreApplication::translate("InputRZForm", "13", nullptr));
        QTableWidgetItem *___qtablewidgetitem631 = raddistTable->verticalHeaderItem(13);
        ___qtablewidgetitem631->setText(QCoreApplication::translate("InputRZForm", "14", nullptr));
        QTableWidgetItem *___qtablewidgetitem632 = raddistTable->verticalHeaderItem(14);
        ___qtablewidgetitem632->setText(QCoreApplication::translate("InputRZForm", "15", nullptr));
        QTableWidgetItem *___qtablewidgetitem633 = raddistTable->verticalHeaderItem(15);
        ___qtablewidgetitem633->setText(QCoreApplication::translate("InputRZForm", "16", nullptr));
        QTableWidgetItem *___qtablewidgetitem634 = raddistTable->verticalHeaderItem(16);
        ___qtablewidgetitem634->setText(QCoreApplication::translate("InputRZForm", "17", nullptr));
        QTableWidgetItem *___qtablewidgetitem635 = raddistTable->verticalHeaderItem(17);
        ___qtablewidgetitem635->setText(QCoreApplication::translate("InputRZForm", "18", nullptr));
        QTableWidgetItem *___qtablewidgetitem636 = raddistTable->verticalHeaderItem(18);
        ___qtablewidgetitem636->setText(QCoreApplication::translate("InputRZForm", "19", nullptr));
        QTableWidgetItem *___qtablewidgetitem637 = raddistTable->verticalHeaderItem(19);
        ___qtablewidgetitem637->setText(QCoreApplication::translate("InputRZForm", "20", nullptr));
        QTableWidgetItem *___qtablewidgetitem638 = raddistTable->verticalHeaderItem(20);
        ___qtablewidgetitem638->setText(QCoreApplication::translate("InputRZForm", "21", nullptr));
        QTableWidgetItem *___qtablewidgetitem639 = raddistTable->verticalHeaderItem(21);
        ___qtablewidgetitem639->setText(QCoreApplication::translate("InputRZForm", "22", nullptr));
        QTableWidgetItem *___qtablewidgetitem640 = raddistTable->verticalHeaderItem(22);
        ___qtablewidgetitem640->setText(QCoreApplication::translate("InputRZForm", "23", nullptr));
        QTableWidgetItem *___qtablewidgetitem641 = raddistTable->verticalHeaderItem(23);
        ___qtablewidgetitem641->setText(QCoreApplication::translate("InputRZForm", "24", nullptr));
        QTableWidgetItem *___qtablewidgetitem642 = raddistTable->verticalHeaderItem(24);
        ___qtablewidgetitem642->setText(QCoreApplication::translate("InputRZForm", "25", nullptr));
        QTableWidgetItem *___qtablewidgetitem643 = raddistTable->verticalHeaderItem(25);
        ___qtablewidgetitem643->setText(QCoreApplication::translate("InputRZForm", "26", nullptr));
        QTableWidgetItem *___qtablewidgetitem644 = raddistTable->verticalHeaderItem(26);
        ___qtablewidgetitem644->setText(QCoreApplication::translate("InputRZForm", "27", nullptr));
        QTableWidgetItem *___qtablewidgetitem645 = raddistTable->verticalHeaderItem(27);
        ___qtablewidgetitem645->setText(QCoreApplication::translate("InputRZForm", "28", nullptr));
        QTableWidgetItem *___qtablewidgetitem646 = raddistTable->verticalHeaderItem(28);
        ___qtablewidgetitem646->setText(QCoreApplication::translate("InputRZForm", "29", nullptr));
        QTableWidgetItem *___qtablewidgetitem647 = raddistTable->verticalHeaderItem(29);
        ___qtablewidgetitem647->setText(QCoreApplication::translate("InputRZForm", "30", nullptr));
        QTableWidgetItem *___qtablewidgetitem648 = raddistTable->verticalHeaderItem(30);
        ___qtablewidgetitem648->setText(QCoreApplication::translate("InputRZForm", "31", nullptr));
        QTableWidgetItem *___qtablewidgetitem649 = raddistTable->verticalHeaderItem(31);
        ___qtablewidgetitem649->setText(QCoreApplication::translate("InputRZForm", "32", nullptr));
        QTableWidgetItem *___qtablewidgetitem650 = raddistTable->verticalHeaderItem(32);
        ___qtablewidgetitem650->setText(QCoreApplication::translate("InputRZForm", "33", nullptr));
        QTableWidgetItem *___qtablewidgetitem651 = raddistTable->verticalHeaderItem(33);
        ___qtablewidgetitem651->setText(QCoreApplication::translate("InputRZForm", "34", nullptr));
        QTableWidgetItem *___qtablewidgetitem652 = raddistTable->verticalHeaderItem(34);
        ___qtablewidgetitem652->setText(QCoreApplication::translate("InputRZForm", "35", nullptr));
        QTableWidgetItem *___qtablewidgetitem653 = raddistTable->verticalHeaderItem(35);
        ___qtablewidgetitem653->setText(QCoreApplication::translate("InputRZForm", "36", nullptr));
        QTableWidgetItem *___qtablewidgetitem654 = raddistTable->verticalHeaderItem(36);
        ___qtablewidgetitem654->setText(QCoreApplication::translate("InputRZForm", "37", nullptr));
        QTableWidgetItem *___qtablewidgetitem655 = raddistTable->verticalHeaderItem(37);
        ___qtablewidgetitem655->setText(QCoreApplication::translate("InputRZForm", "38", nullptr));
        QTableWidgetItem *___qtablewidgetitem656 = raddistTable->verticalHeaderItem(38);
        ___qtablewidgetitem656->setText(QCoreApplication::translate("InputRZForm", "39", nullptr));
        QTableWidgetItem *___qtablewidgetitem657 = raddistTable->verticalHeaderItem(39);
        ___qtablewidgetitem657->setText(QCoreApplication::translate("InputRZForm", "40", nullptr));
        QTableWidgetItem *___qtablewidgetitem658 = raddistTable->verticalHeaderItem(40);
        ___qtablewidgetitem658->setText(QCoreApplication::translate("InputRZForm", "41", nullptr));
        QTableWidgetItem *___qtablewidgetitem659 = raddistTable->verticalHeaderItem(41);
        ___qtablewidgetitem659->setText(QCoreApplication::translate("InputRZForm", "42", nullptr));
        QTableWidgetItem *___qtablewidgetitem660 = raddistTable->verticalHeaderItem(42);
        ___qtablewidgetitem660->setText(QCoreApplication::translate("InputRZForm", "43", nullptr));
        QTableWidgetItem *___qtablewidgetitem661 = raddistTable->verticalHeaderItem(43);
        ___qtablewidgetitem661->setText(QCoreApplication::translate("InputRZForm", "44", nullptr));
        QTableWidgetItem *___qtablewidgetitem662 = raddistTable->verticalHeaderItem(44);
        ___qtablewidgetitem662->setText(QCoreApplication::translate("InputRZForm", "45", nullptr));
        QTableWidgetItem *___qtablewidgetitem663 = raddistTable->verticalHeaderItem(45);
        ___qtablewidgetitem663->setText(QCoreApplication::translate("InputRZForm", "46", nullptr));
        QTableWidgetItem *___qtablewidgetitem664 = raddistTable->verticalHeaderItem(46);
        ___qtablewidgetitem664->setText(QCoreApplication::translate("InputRZForm", "47", nullptr));
        QTableWidgetItem *___qtablewidgetitem665 = raddistTable->verticalHeaderItem(47);
        ___qtablewidgetitem665->setText(QCoreApplication::translate("InputRZForm", "48", nullptr));
        QTableWidgetItem *___qtablewidgetitem666 = raddistTable->verticalHeaderItem(48);
        ___qtablewidgetitem666->setText(QCoreApplication::translate("InputRZForm", "49", nullptr));
        QTableWidgetItem *___qtablewidgetitem667 = raddistTable->verticalHeaderItem(49);
        ___qtablewidgetitem667->setText(QCoreApplication::translate("InputRZForm", "50", nullptr));
        raddistfnameGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "radial distribution file name", nullptr));
        raddistfnameComboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "dummy.file", nullptr));

        raddistfnameButton->setText(QCoreApplication::translate("InputRZForm", "...", nullptr));
        ioutrdistGroupBox->setTitle(QString());
        ioutrdistCheckBox->setText(QCoreApplication::translate("InputRZForm", "Include distrib. data in summary", nullptr));
        TabWidgetRZ->setTabText(TabWidgetRZ->indexOf(SItab), QCoreApplication::translate("InputRZForm", "Source", nullptr));
#if QT_CONFIG(tooltip)
        PairAngSamplingGroupBox->setToolTip(QCoreApplication::translate("InputRZForm", "If off, pairs set in motion at angle m/E. If Simple, leading term of angular dist. turned on. If KM, turns on 2BS from Koch and Motz", nullptr));
#endif // QT_CONFIG(tooltip)
        PairAngSamplingGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Pair angular sampling", nullptr));
        PairAngSamplingComboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "Simple", nullptr));
        PairAngSamplingComboBox->setItemText(1, QCoreApplication::translate("InputRZForm", "Off", nullptr));
        PairAngSamplingComboBox->setItemText(2, QCoreApplication::translate("InputRZForm", "KM", nullptr));

#if QT_CONFIG(tooltip)
        BremsAngSamplingGroupBox->setToolTip(QCoreApplication::translate("InputRZForm", "If KM, complete modified Koch-Motz 2BS used. If Simple, only leading term of Koch-Motz distribution used.", nullptr));
#endif // QT_CONFIG(tooltip)
        BremsAngSamplingGroupBox->setTitle(QCoreApplication::translate("InputRZForm", " Brems angular sampling", nullptr));
        BremsAngSamplingComboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "KM", nullptr));
        BremsAngSamplingComboBox->setItemText(1, QCoreApplication::translate("InputRZForm", "Simple", nullptr));

#if QT_CONFIG(tooltip)
        BremsXSectionGroupBox->setToolTip(QCoreApplication::translate("InputRZForm", "If BH,  Bethe-Heitler brems. cross sections used. If NIST, NIST brems cross section used (basis for ICRU radiative stopping powers).", nullptr));
#endif // QT_CONFIG(tooltip)
        BremsXSectionGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Brems cross sections", nullptr));
        BremsXSectionComboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "BH", nullptr));
        BremsXSectionComboBox->setItemText(1, QCoreApplication::translate("InputRZForm", "NIST", nullptr));

#if QT_CONFIG(tooltip)
        estep_algorithmGroupBox->setToolTip(QCoreApplication::translate("InputRZForm", "Algorithm used to take into account lateral and longitudinal correlations in a condensed history step.", nullptr));
#endif // QT_CONFIG(tooltip)
        estep_algorithmGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Electron-step algorithm", nullptr));
        estep_algorithmComboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "PRESTA-II", nullptr));
        estep_algorithmComboBox->setItemText(1, QCoreApplication::translate("InputRZForm", "PRESTA-I", nullptr));

        EIIgroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Electron impact ionization", nullptr));
        EIIcomboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "On", nullptr));
        EIIcomboBox->setItemText(1, QCoreApplication::translate("InputRZForm", "Off", nullptr));
        EIIcomboBox->setItemText(2, QCoreApplication::translate("InputRZForm", "CASNATI", nullptr));
        EIIcomboBox->setItemText(3, QCoreApplication::translate("InputRZForm", "KOLBENSTVEDT", nullptr));
        EIIcomboBox->setItemText(4, QCoreApplication::translate("InputRZForm", "GRYZINSKI", nullptr));

#if QT_CONFIG(tooltip)
        BCAGroupBox->setToolTip(QCoreApplication::translate("InputRZForm", "Two selections possible: EXACT, means crossing boundaries in a single scattering mode and PRESTA-I, boundaries crossed a la PRESTA, i.e. lateral correlations turned off and MS forced at boundaries. ", nullptr));
#endif // QT_CONFIG(tooltip)
        BCAGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Boundary crossing algorithm", nullptr));
        BCAComboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "EXACT", nullptr));
        BCAComboBox->setItemText(1, QCoreApplication::translate("InputRZForm", "PRESTA-I", nullptr));

#if QT_CONFIG(tooltip)
        GECUTGroupBox->setToolTip(QCoreApplication::translate("InputRZForm", "Global electron transport cut off energy (in MeV, includes electron rest energy).", nullptr));
#endif // QT_CONFIG(tooltip)
        GECUTGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Global ECUT", nullptr));
        GECUTEdit->setText(QCoreApplication::translate("InputRZForm", "0.521", nullptr));
#if QT_CONFIG(tooltip)
        GPCUTGroupBox->setToolTip(QCoreApplication::translate("InputRZForm", "Global photon transport cut off energy (in MeV).", nullptr));
#endif // QT_CONFIG(tooltip)
        GPCUTGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Global PCUT", nullptr));
        GPCUTEdit->setText(QCoreApplication::translate("InputRZForm", "0.001", nullptr));
#if QT_CONFIG(tooltip)
        GSMAXGroupBox->setToolTip(QCoreApplication::translate("InputRZForm", "Global maximum step-size restriction for electron transport (in cm). If using default EGSnrc electron-step algorithm, no SMAX-restriction necessary.", nullptr));
#endif // QT_CONFIG(tooltip)
        GSMAXGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Global SMAX", nullptr));
        RayleighgroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Rayleigh scattering", nullptr));
        RayleighcomboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "Off", nullptr));
        RayleighcomboBox->setItemText(1, QCoreApplication::translate("InputRZForm", "On", nullptr));
        RayleighcomboBox->setItemText(2, QCoreApplication::translate("InputRZForm", "custom", nullptr));
        RayleighcomboBox->setItemText(3, QCoreApplication::translate("InputRZForm", "On in Regions", nullptr));
        RayleighcomboBox->setItemText(4, QCoreApplication::translate("InputRZForm", "Off in Regions", nullptr));

        PEgroupBox->setTitle(QCoreApplication::translate("InputRZForm", "PE angular sampling", nullptr));
        PEcomboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "On", nullptr));
        PEcomboBox->setItemText(1, QCoreApplication::translate("InputRZForm", "Off", nullptr));
        PEcomboBox->setItemText(2, QCoreApplication::translate("InputRZForm", "On in Regions", nullptr));
        PEcomboBox->setItemText(3, QCoreApplication::translate("InputRZForm", "Off in Regions", nullptr));

        BoundComptongroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Bound Compton", nullptr));
        BoundComptoncomboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "On", nullptr));
        BoundComptoncomboBox->setItemText(1, QCoreApplication::translate("InputRZForm", "Off", nullptr));
        BoundComptoncomboBox->setItemText(2, QCoreApplication::translate("InputRZForm", "norej", nullptr));
        BoundComptoncomboBox->setItemText(3, QCoreApplication::translate("InputRZForm", "simple", nullptr));
        BoundComptoncomboBox->setItemText(4, QCoreApplication::translate("InputRZForm", "On in Regions", nullptr));
        BoundComptoncomboBox->setItemText(5, QCoreApplication::translate("InputRZForm", "Off in Regions", nullptr));

        RelaxationgroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Atomic relaxations", nullptr));
        RelaxationcomboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "On", nullptr));
        RelaxationcomboBox->setItemText(1, QCoreApplication::translate("InputRZForm", "Off", nullptr));
        RelaxationcomboBox->setItemText(2, QCoreApplication::translate("InputRZForm", "On in Regions", nullptr));
        RelaxationcomboBox->setItemText(3, QCoreApplication::translate("InputRZForm", "Off in Regions", nullptr));
        RelaxationcomboBox->setItemText(4, QCoreApplication::translate("InputRZForm", "eadl", nullptr));
        RelaxationcomboBox->setItemText(5, QCoreApplication::translate("InputRZForm", "simple", nullptr));

        ECUTCheckBox->setText(QCoreApplication::translate("InputRZForm", "ECUT by regions", nullptr));
        PCUTCheckBox->setText(QCoreApplication::translate("InputRZForm", "PCUT by regions", nullptr));
        SMAXCheckBox->setText(QCoreApplication::translate("InputRZForm", "SMAX by regions", nullptr));
        photonXSectiongroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Photon cross section library", nullptr));
        photonXSectioncomboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "PEGS4", nullptr));
        photonXSectioncomboBox->setItemText(1, QCoreApplication::translate("InputRZForm", "XCOM", nullptr));
        photonXSectioncomboBox->setItemText(2, QCoreApplication::translate("InputRZForm", "EPDL", nullptr));

#if QT_CONFIG(tooltip)
        photonXSectionOutCheckBox->setToolTip(QCoreApplication::translate("InputRZForm", "If On, then a file $EGS_HOME/user_code/inputfile.xsections is output containing photon cross-section data used.", nullptr));
#endif // QT_CONFIG(tooltip)
        photonXSectionOutCheckBox->setText(QCoreApplication::translate("InputRZForm", "Photon cross-sections output", nullptr));
        customFFgroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Custom Rayleigh Form Factors:", nullptr));
        QTableWidgetItem *___qtablewidgetitem668 = customFFTable->horizontalHeaderItem(0);
        ___qtablewidgetitem668->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem669 = customFFTable->horizontalHeaderItem(1);
        ___qtablewidgetitem669->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem670 = customFFTable->verticalHeaderItem(0);
        ___qtablewidgetitem670->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem671 = customFFTable->verticalHeaderItem(1);
        ___qtablewidgetitem671->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem672 = customFFTable->verticalHeaderItem(2);
        ___qtablewidgetitem672->setText(QCoreApplication::translate("InputRZForm", "3", nullptr));
        QTableWidgetItem *___qtablewidgetitem673 = customFFTable->verticalHeaderItem(3);
        ___qtablewidgetitem673->setText(QCoreApplication::translate("InputRZForm", "4", nullptr));
        QTableWidgetItem *___qtablewidgetitem674 = customFFTable->verticalHeaderItem(4);
        ___qtablewidgetitem674->setText(QCoreApplication::translate("InputRZForm", "5", nullptr));
        QTableWidgetItem *___qtablewidgetitem675 = customFFTable->verticalHeaderItem(5);
        ___qtablewidgetitem675->setText(QCoreApplication::translate("InputRZForm", "6", nullptr));
        QTableWidgetItem *___qtablewidgetitem676 = customFFTable->verticalHeaderItem(6);
        ___qtablewidgetitem676->setText(QCoreApplication::translate("InputRZForm", "7", nullptr));
        QTableWidgetItem *___qtablewidgetitem677 = customFFTable->verticalHeaderItem(7);
        ___qtablewidgetitem677->setText(QCoreApplication::translate("InputRZForm", "8", nullptr));
        QTableWidgetItem *___qtablewidgetitem678 = customFFTable->verticalHeaderItem(8);
        ___qtablewidgetitem678->setText(QCoreApplication::translate("InputRZForm", "9", nullptr));
        QTableWidgetItem *___qtablewidgetitem679 = customFFTable->verticalHeaderItem(9);
        ___qtablewidgetitem679->setText(QCoreApplication::translate("InputRZForm", "10", nullptr));
        QTableWidgetItem *___qtablewidgetitem680 = customFFTable->verticalHeaderItem(10);
        ___qtablewidgetitem680->setText(QCoreApplication::translate("InputRZForm", "11", nullptr));
        QTableWidgetItem *___qtablewidgetitem681 = customFFTable->verticalHeaderItem(11);
        ___qtablewidgetitem681->setText(QCoreApplication::translate("InputRZForm", "12", nullptr));
        QTableWidgetItem *___qtablewidgetitem682 = customFFTable->verticalHeaderItem(12);
        ___qtablewidgetitem682->setText(QCoreApplication::translate("InputRZForm", "13", nullptr));
        QTableWidgetItem *___qtablewidgetitem683 = customFFTable->verticalHeaderItem(13);
        ___qtablewidgetitem683->setText(QCoreApplication::translate("InputRZForm", "14", nullptr));
        QTableWidgetItem *___qtablewidgetitem684 = customFFTable->verticalHeaderItem(14);
        ___qtablewidgetitem684->setText(QCoreApplication::translate("InputRZForm", "15", nullptr));
        QTableWidgetItem *___qtablewidgetitem685 = customFFTable->verticalHeaderItem(15);
        ___qtablewidgetitem685->setText(QCoreApplication::translate("InputRZForm", "16", nullptr));
        QTableWidgetItem *___qtablewidgetitem686 = customFFTable->verticalHeaderItem(16);
        ___qtablewidgetitem686->setText(QCoreApplication::translate("InputRZForm", "17", nullptr));
        QTableWidgetItem *___qtablewidgetitem687 = customFFTable->verticalHeaderItem(17);
        ___qtablewidgetitem687->setText(QCoreApplication::translate("InputRZForm", "18", nullptr));
        QTableWidgetItem *___qtablewidgetitem688 = customFFTable->verticalHeaderItem(18);
        ___qtablewidgetitem688->setText(QCoreApplication::translate("InputRZForm", "19", nullptr));
        QTableWidgetItem *___qtablewidgetitem689 = customFFTable->verticalHeaderItem(19);
        ___qtablewidgetitem689->setText(QCoreApplication::translate("InputRZForm", "20", nullptr));
        QTableWidgetItem *___qtablewidgetitem690 = customFFTable->verticalHeaderItem(20);
        ___qtablewidgetitem690->setText(QCoreApplication::translate("InputRZForm", "21", nullptr));
        QTableWidgetItem *___qtablewidgetitem691 = customFFTable->verticalHeaderItem(21);
        ___qtablewidgetitem691->setText(QCoreApplication::translate("InputRZForm", "22", nullptr));
        QTableWidgetItem *___qtablewidgetitem692 = customFFTable->verticalHeaderItem(22);
        ___qtablewidgetitem692->setText(QCoreApplication::translate("InputRZForm", "23", nullptr));
        QTableWidgetItem *___qtablewidgetitem693 = customFFTable->verticalHeaderItem(23);
        ___qtablewidgetitem693->setText(QCoreApplication::translate("InputRZForm", "24", nullptr));
        QTableWidgetItem *___qtablewidgetitem694 = customFFTable->verticalHeaderItem(24);
        ___qtablewidgetitem694->setText(QCoreApplication::translate("InputRZForm", "25", nullptr));
        QTableWidgetItem *___qtablewidgetitem695 = customFFTable->verticalHeaderItem(25);
        ___qtablewidgetitem695->setText(QCoreApplication::translate("InputRZForm", "26", nullptr));
        QTableWidgetItem *___qtablewidgetitem696 = customFFTable->verticalHeaderItem(26);
        ___qtablewidgetitem696->setText(QCoreApplication::translate("InputRZForm", "27", nullptr));
        QTableWidgetItem *___qtablewidgetitem697 = customFFTable->verticalHeaderItem(27);
        ___qtablewidgetitem697->setText(QCoreApplication::translate("InputRZForm", "28", nullptr));
        QTableWidgetItem *___qtablewidgetitem698 = customFFTable->verticalHeaderItem(28);
        ___qtablewidgetitem698->setText(QCoreApplication::translate("InputRZForm", "29", nullptr));
        QTableWidgetItem *___qtablewidgetitem699 = customFFTable->verticalHeaderItem(29);
        ___qtablewidgetitem699->setText(QCoreApplication::translate("InputRZForm", "30", nullptr));
        QTableWidgetItem *___qtablewidgetitem700 = customFFTable->verticalHeaderItem(30);
        ___qtablewidgetitem700->setText(QCoreApplication::translate("InputRZForm", "31", nullptr));
        QTableWidgetItem *___qtablewidgetitem701 = customFFTable->verticalHeaderItem(31);
        ___qtablewidgetitem701->setText(QCoreApplication::translate("InputRZForm", "32", nullptr));
        QTableWidgetItem *___qtablewidgetitem702 = customFFTable->verticalHeaderItem(32);
        ___qtablewidgetitem702->setText(QCoreApplication::translate("InputRZForm", "33", nullptr));
        QTableWidgetItem *___qtablewidgetitem703 = customFFTable->verticalHeaderItem(33);
        ___qtablewidgetitem703->setText(QCoreApplication::translate("InputRZForm", "34", nullptr));
        QTableWidgetItem *___qtablewidgetitem704 = customFFTable->verticalHeaderItem(34);
        ___qtablewidgetitem704->setText(QCoreApplication::translate("InputRZForm", "35", nullptr));
        QTableWidgetItem *___qtablewidgetitem705 = customFFTable->verticalHeaderItem(35);
        ___qtablewidgetitem705->setText(QCoreApplication::translate("InputRZForm", "36", nullptr));
        QTableWidgetItem *___qtablewidgetitem706 = customFFTable->verticalHeaderItem(36);
        ___qtablewidgetitem706->setText(QCoreApplication::translate("InputRZForm", "37", nullptr));
        QTableWidgetItem *___qtablewidgetitem707 = customFFTable->verticalHeaderItem(37);
        ___qtablewidgetitem707->setText(QCoreApplication::translate("InputRZForm", "38", nullptr));
        QTableWidgetItem *___qtablewidgetitem708 = customFFTable->verticalHeaderItem(38);
        ___qtablewidgetitem708->setText(QCoreApplication::translate("InputRZForm", "39", nullptr));
        QTableWidgetItem *___qtablewidgetitem709 = customFFTable->verticalHeaderItem(39);
        ___qtablewidgetitem709->setText(QCoreApplication::translate("InputRZForm", "40", nullptr));
        QTableWidgetItem *___qtablewidgetitem710 = customFFTable->verticalHeaderItem(40);
        ___qtablewidgetitem710->setText(QCoreApplication::translate("InputRZForm", "41", nullptr));
        QTableWidgetItem *___qtablewidgetitem711 = customFFTable->verticalHeaderItem(41);
        ___qtablewidgetitem711->setText(QCoreApplication::translate("InputRZForm", "42", nullptr));
        QTableWidgetItem *___qtablewidgetitem712 = customFFTable->verticalHeaderItem(42);
        ___qtablewidgetitem712->setText(QCoreApplication::translate("InputRZForm", "43", nullptr));
        QTableWidgetItem *___qtablewidgetitem713 = customFFTable->verticalHeaderItem(43);
        ___qtablewidgetitem713->setText(QCoreApplication::translate("InputRZForm", "44", nullptr));
        QTableWidgetItem *___qtablewidgetitem714 = customFFTable->verticalHeaderItem(44);
        ___qtablewidgetitem714->setText(QCoreApplication::translate("InputRZForm", "45", nullptr));
        QTableWidgetItem *___qtablewidgetitem715 = customFFTable->verticalHeaderItem(45);
        ___qtablewidgetitem715->setText(QCoreApplication::translate("InputRZForm", "46", nullptr));
        QTableWidgetItem *___qtablewidgetitem716 = customFFTable->verticalHeaderItem(46);
        ___qtablewidgetitem716->setText(QCoreApplication::translate("InputRZForm", "47", nullptr));
        QTableWidgetItem *___qtablewidgetitem717 = customFFTable->verticalHeaderItem(47);
        ___qtablewidgetitem717->setText(QCoreApplication::translate("InputRZForm", "48", nullptr));
        QTableWidgetItem *___qtablewidgetitem718 = customFFTable->verticalHeaderItem(48);
        ___qtablewidgetitem718->setText(QCoreApplication::translate("InputRZForm", "49", nullptr));
        QTableWidgetItem *___qtablewidgetitem719 = customFFTable->verticalHeaderItem(49);
        ___qtablewidgetitem719->setText(QCoreApplication::translate("InputRZForm", "50", nullptr));
#if QT_CONFIG(tooltip)
        ESTEPELabel->setToolTip(QCoreApplication::translate("InputRZForm", "Maximum fractional energy loss per step. Default is 0.25 (25%).", nullptr));
#endif // QT_CONFIG(tooltip)
        ESTEPELabel->setText(QCoreApplication::translate("InputRZForm", "ESTEPE", nullptr));
#if QT_CONFIG(tooltip)
        ESTEPEEdit->setToolTip(QCoreApplication::translate("InputRZForm", "Maximum fractional energy loss per step. Default is 0.25 (25%).", nullptr));
#endif // QT_CONFIG(tooltip)
        ESTEPEEdit->setText(QCoreApplication::translate("InputRZForm", "0.25", nullptr));
#if QT_CONFIG(tooltip)
        XImaxLabel->setToolTip(QCoreApplication::translate("InputRZForm", "Maximum first elastic scattering moment per step. NEVER use value greater than 1 as this is beyond the range of MS data available.", nullptr));
#endif // QT_CONFIG(tooltip)
        XImaxLabel->setText(QCoreApplication::translate("InputRZForm", "XImax", nullptr));
#if QT_CONFIG(tooltip)
        XImaxEdit->setToolTip(QCoreApplication::translate("InputRZForm", "Maximum first elastic scattering moment per step. NEVER use value greater than 1 as this is beyond the range of MS data available.", nullptr));
#endif // QT_CONFIG(tooltip)
        XImaxEdit->setText(QCoreApplication::translate("InputRZForm", "0.5", nullptr));
#if QT_CONFIG(tooltip)
        SkinDepthLabel->setToolTip(QCoreApplication::translate("InputRZForm", "distance from a boundary at which the algorithm will go into single scattering mode (in elastic MFP)", nullptr));
#endif // QT_CONFIG(tooltip)
        SkinDepthLabel->setText(QCoreApplication::translate("InputRZForm", "Skin depth for BCA", nullptr));
#if QT_CONFIG(tooltip)
        SkinDepthEdit->setToolTip(QCoreApplication::translate("InputRZForm", "distance from a boundary at which the algorithm will go into single scattering mode (in elastic MFP)", nullptr));
#endif // QT_CONFIG(tooltip)
        SkinDepthEdit->setText(QCoreApplication::translate("InputRZForm", "3", nullptr));
#if QT_CONFIG(tooltip)
        SpinCheckBox->setToolTip(QCoreApplication::translate("InputRZForm", "Turns off/on spin effects for electron elastic scattering. ABSOLUTELY necessary for good backscattering calculations.", nullptr));
#endif // QT_CONFIG(tooltip)
        SpinCheckBox->setText(QCoreApplication::translate("InputRZForm", "Spin effects", nullptr));
        TabWidgetRZ->setTabText(TabWidgetRZ->indexOf(MCTPtab), QCoreApplication::translate("InputRZForm", "Transport Parameters", nullptr));
        PhotonForcingGroupBox->setTitle(QString());
        PhotonForcingCheckBox->setText(QCoreApplication::translate("InputRZForm", "Photon Forcing", nullptr));
        StartForcingLabel->setText(QCoreApplication::translate("InputRZForm", "start forcing", nullptr));
        StopForcingLabel->setText(QCoreApplication::translate("InputRZForm", "stop forcing", nullptr));
        ExpTrafoCGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Photon pathlength biasing", nullptr));
        ExpTrafoCLabel->setText(QCoreApplication::translate("InputRZForm", "biasing parameter C:", nullptr));
        ExpTrafoCEdit->setText(QCoreApplication::translate("InputRZForm", "0.0", nullptr));
        RRGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Russian Roulette (photons)", nullptr));
        RRDepthLabel->setText(QCoreApplication::translate("InputRZForm", "Z Depth:", nullptr));
        RRDepthEdit->setText(QCoreApplication::translate("InputRZForm", "0.0", nullptr));
        RRFractionLabel->setText(QCoreApplication::translate("InputRZForm", "Fraction:", nullptr));
        RRFractionEdit->setText(QCoreApplication::translate("InputRZForm", "0.0", nullptr));
        photonSplitGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Photon splitting", nullptr));
        photonSplitLabel->setText(QCoreApplication::translate("InputRZForm", "splitting number", nullptr));
        CSEnhancementGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Photon cross section enhancement", nullptr));
        TextLabel8->setText(QCoreApplication::translate("InputRZForm", "CS Enhancement factor", nullptr));
        CSEnhancement_RegionsLabel1->setText(QCoreApplication::translate("InputRZForm", "Regions to apply CS Enhancement", nullptr));
        CSEnhancement_RegionsLabel2->setText(QCoreApplication::translate("InputRZForm", " (DOSRZnrc only)", nullptr));
        QTableWidgetItem *___qtablewidgetitem720 = CSEnhancementTable->horizontalHeaderItem(0);
        ___qtablewidgetitem720->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem721 = CSEnhancementTable->horizontalHeaderItem(1);
        ___qtablewidgetitem721->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem722 = CSEnhancementTable->verticalHeaderItem(0);
        ___qtablewidgetitem722->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem723 = CSEnhancementTable->verticalHeaderItem(1);
        ___qtablewidgetitem723->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem724 = CSEnhancementTable->verticalHeaderItem(2);
        ___qtablewidgetitem724->setText(QCoreApplication::translate("InputRZForm", "3", nullptr));
        QTableWidgetItem *___qtablewidgetitem725 = CSEnhancementTable->verticalHeaderItem(3);
        ___qtablewidgetitem725->setText(QCoreApplication::translate("InputRZForm", "4", nullptr));
        QTableWidgetItem *___qtablewidgetitem726 = CSEnhancementTable->verticalHeaderItem(4);
        ___qtablewidgetitem726->setText(QCoreApplication::translate("InputRZForm", "5", nullptr));
        QTableWidgetItem *___qtablewidgetitem727 = CSEnhancementTable->verticalHeaderItem(5);
        ___qtablewidgetitem727->setText(QCoreApplication::translate("InputRZForm", "6", nullptr));
        QTableWidgetItem *___qtablewidgetitem728 = CSEnhancementTable->verticalHeaderItem(6);
        ___qtablewidgetitem728->setText(QCoreApplication::translate("InputRZForm", "7", nullptr));
        QTableWidgetItem *___qtablewidgetitem729 = CSEnhancementTable->verticalHeaderItem(7);
        ___qtablewidgetitem729->setText(QCoreApplication::translate("InputRZForm", "8", nullptr));
        QTableWidgetItem *___qtablewidgetitem730 = CSEnhancementTable->verticalHeaderItem(8);
        ___qtablewidgetitem730->setText(QCoreApplication::translate("InputRZForm", "9", nullptr));
        QTableWidgetItem *___qtablewidgetitem731 = CSEnhancementTable->verticalHeaderItem(9);
        ___qtablewidgetitem731->setText(QCoreApplication::translate("InputRZForm", "10", nullptr));
        QTableWidgetItem *___qtablewidgetitem732 = CSEnhancementTable->verticalHeaderItem(10);
        ___qtablewidgetitem732->setText(QCoreApplication::translate("InputRZForm", "11", nullptr));
        QTableWidgetItem *___qtablewidgetitem733 = CSEnhancementTable->verticalHeaderItem(11);
        ___qtablewidgetitem733->setText(QCoreApplication::translate("InputRZForm", "12", nullptr));
        QTableWidgetItem *___qtablewidgetitem734 = CSEnhancementTable->verticalHeaderItem(12);
        ___qtablewidgetitem734->setText(QCoreApplication::translate("InputRZForm", "13", nullptr));
        QTableWidgetItem *___qtablewidgetitem735 = CSEnhancementTable->verticalHeaderItem(13);
        ___qtablewidgetitem735->setText(QCoreApplication::translate("InputRZForm", "14", nullptr));
        QTableWidgetItem *___qtablewidgetitem736 = CSEnhancementTable->verticalHeaderItem(14);
        ___qtablewidgetitem736->setText(QCoreApplication::translate("InputRZForm", "15", nullptr));
        QTableWidgetItem *___qtablewidgetitem737 = CSEnhancementTable->verticalHeaderItem(15);
        ___qtablewidgetitem737->setText(QCoreApplication::translate("InputRZForm", "16", nullptr));
        QTableWidgetItem *___qtablewidgetitem738 = CSEnhancementTable->verticalHeaderItem(16);
        ___qtablewidgetitem738->setText(QCoreApplication::translate("InputRZForm", "17", nullptr));
        QTableWidgetItem *___qtablewidgetitem739 = CSEnhancementTable->verticalHeaderItem(17);
        ___qtablewidgetitem739->setText(QCoreApplication::translate("InputRZForm", "18", nullptr));
        QTableWidgetItem *___qtablewidgetitem740 = CSEnhancementTable->verticalHeaderItem(18);
        ___qtablewidgetitem740->setText(QCoreApplication::translate("InputRZForm", "19", nullptr));
        QTableWidgetItem *___qtablewidgetitem741 = CSEnhancementTable->verticalHeaderItem(19);
        ___qtablewidgetitem741->setText(QCoreApplication::translate("InputRZForm", "20", nullptr));
        QTableWidgetItem *___qtablewidgetitem742 = CSEnhancementTable->verticalHeaderItem(20);
        ___qtablewidgetitem742->setText(QCoreApplication::translate("InputRZForm", "21", nullptr));
        QTableWidgetItem *___qtablewidgetitem743 = CSEnhancementTable->verticalHeaderItem(21);
        ___qtablewidgetitem743->setText(QCoreApplication::translate("InputRZForm", "22", nullptr));
        QTableWidgetItem *___qtablewidgetitem744 = CSEnhancementTable->verticalHeaderItem(22);
        ___qtablewidgetitem744->setText(QCoreApplication::translate("InputRZForm", "23", nullptr));
        QTableWidgetItem *___qtablewidgetitem745 = CSEnhancementTable->verticalHeaderItem(23);
        ___qtablewidgetitem745->setText(QCoreApplication::translate("InputRZForm", "24", nullptr));
        QTableWidgetItem *___qtablewidgetitem746 = CSEnhancementTable->verticalHeaderItem(24);
        ___qtablewidgetitem746->setText(QCoreApplication::translate("InputRZForm", "25", nullptr));
        QTableWidgetItem *___qtablewidgetitem747 = CSEnhancementTable->verticalHeaderItem(25);
        ___qtablewidgetitem747->setText(QCoreApplication::translate("InputRZForm", "26", nullptr));
        QTableWidgetItem *___qtablewidgetitem748 = CSEnhancementTable->verticalHeaderItem(26);
        ___qtablewidgetitem748->setText(QCoreApplication::translate("InputRZForm", "27", nullptr));
        QTableWidgetItem *___qtablewidgetitem749 = CSEnhancementTable->verticalHeaderItem(27);
        ___qtablewidgetitem749->setText(QCoreApplication::translate("InputRZForm", "28", nullptr));
        QTableWidgetItem *___qtablewidgetitem750 = CSEnhancementTable->verticalHeaderItem(28);
        ___qtablewidgetitem750->setText(QCoreApplication::translate("InputRZForm", "29", nullptr));
        QTableWidgetItem *___qtablewidgetitem751 = CSEnhancementTable->verticalHeaderItem(29);
        ___qtablewidgetitem751->setText(QCoreApplication::translate("InputRZForm", "30", nullptr));
        QTableWidgetItem *___qtablewidgetitem752 = CSEnhancementTable->verticalHeaderItem(30);
        ___qtablewidgetitem752->setText(QCoreApplication::translate("InputRZForm", "31", nullptr));
        QTableWidgetItem *___qtablewidgetitem753 = CSEnhancementTable->verticalHeaderItem(31);
        ___qtablewidgetitem753->setText(QCoreApplication::translate("InputRZForm", "32", nullptr));
        QTableWidgetItem *___qtablewidgetitem754 = CSEnhancementTable->verticalHeaderItem(32);
        ___qtablewidgetitem754->setText(QCoreApplication::translate("InputRZForm", "33", nullptr));
        QTableWidgetItem *___qtablewidgetitem755 = CSEnhancementTable->verticalHeaderItem(33);
        ___qtablewidgetitem755->setText(QCoreApplication::translate("InputRZForm", "34", nullptr));
        QTableWidgetItem *___qtablewidgetitem756 = CSEnhancementTable->verticalHeaderItem(34);
        ___qtablewidgetitem756->setText(QCoreApplication::translate("InputRZForm", "35", nullptr));
        QTableWidgetItem *___qtablewidgetitem757 = CSEnhancementTable->verticalHeaderItem(35);
        ___qtablewidgetitem757->setText(QCoreApplication::translate("InputRZForm", "36", nullptr));
        QTableWidgetItem *___qtablewidgetitem758 = CSEnhancementTable->verticalHeaderItem(36);
        ___qtablewidgetitem758->setText(QCoreApplication::translate("InputRZForm", "37", nullptr));
        QTableWidgetItem *___qtablewidgetitem759 = CSEnhancementTable->verticalHeaderItem(37);
        ___qtablewidgetitem759->setText(QCoreApplication::translate("InputRZForm", "38", nullptr));
        QTableWidgetItem *___qtablewidgetitem760 = CSEnhancementTable->verticalHeaderItem(38);
        ___qtablewidgetitem760->setText(QCoreApplication::translate("InputRZForm", "39", nullptr));
        QTableWidgetItem *___qtablewidgetitem761 = CSEnhancementTable->verticalHeaderItem(39);
        ___qtablewidgetitem761->setText(QCoreApplication::translate("InputRZForm", "40", nullptr));
        QTableWidgetItem *___qtablewidgetitem762 = CSEnhancementTable->verticalHeaderItem(40);
        ___qtablewidgetitem762->setText(QCoreApplication::translate("InputRZForm", "41", nullptr));
        QTableWidgetItem *___qtablewidgetitem763 = CSEnhancementTable->verticalHeaderItem(41);
        ___qtablewidgetitem763->setText(QCoreApplication::translate("InputRZForm", "42", nullptr));
        QTableWidgetItem *___qtablewidgetitem764 = CSEnhancementTable->verticalHeaderItem(42);
        ___qtablewidgetitem764->setText(QCoreApplication::translate("InputRZForm", "43", nullptr));
        QTableWidgetItem *___qtablewidgetitem765 = CSEnhancementTable->verticalHeaderItem(43);
        ___qtablewidgetitem765->setText(QCoreApplication::translate("InputRZForm", "44", nullptr));
        QTableWidgetItem *___qtablewidgetitem766 = CSEnhancementTable->verticalHeaderItem(44);
        ___qtablewidgetitem766->setText(QCoreApplication::translate("InputRZForm", "45", nullptr));
        QTableWidgetItem *___qtablewidgetitem767 = CSEnhancementTable->verticalHeaderItem(45);
        ___qtablewidgetitem767->setText(QCoreApplication::translate("InputRZForm", "46", nullptr));
        QTableWidgetItem *___qtablewidgetitem768 = CSEnhancementTable->verticalHeaderItem(46);
        ___qtablewidgetitem768->setText(QCoreApplication::translate("InputRZForm", "47", nullptr));
        QTableWidgetItem *___qtablewidgetitem769 = CSEnhancementTable->verticalHeaderItem(47);
        ___qtablewidgetitem769->setText(QCoreApplication::translate("InputRZForm", "48", nullptr));
        QTableWidgetItem *___qtablewidgetitem770 = CSEnhancementTable->verticalHeaderItem(48);
        ___qtablewidgetitem770->setText(QCoreApplication::translate("InputRZForm", "49", nullptr));
        QTableWidgetItem *___qtablewidgetitem771 = CSEnhancementTable->verticalHeaderItem(49);
        ___qtablewidgetitem771->setText(QCoreApplication::translate("InputRZForm", "50", nullptr));
        ESAVEINGroupBox->setTitle(QString());
        eRangeRejCheckBox->setText(QCoreApplication::translate("InputRZForm", "Electron Range Rejection", nullptr));
        ESAVEINLabel->setText(QCoreApplication::translate("InputRZForm", "ESAVEIN", nullptr));
        BremsSplitGroupBox->setTitle(QString());
        BremsSplitTextLabel->setText(QCoreApplication::translate("InputRZForm", "# of brems. per event", nullptr));
        BremsSplitCheckBox->setText(QCoreApplication::translate("InputRZForm", "Bremsstrahlung splitting", nullptr));
        ChargedPartRRCheckBox->setText(QCoreApplication::translate("InputRZForm", "Russian Roulette (charged particles )", nullptr));
        TabWidgetRZ->setTabText(TabWidgetRZ->indexOf(VRtab), QCoreApplication::translate("InputRZForm", "Variance Reduction", nullptr));
        PCUTGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "PCUT by regions", nullptr));
        QTableWidgetItem *___qtablewidgetitem772 = PCUTTable->horizontalHeaderItem(0);
        ___qtablewidgetitem772->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem773 = PCUTTable->horizontalHeaderItem(1);
        ___qtablewidgetitem773->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem774 = PCUTTable->horizontalHeaderItem(2);
        ___qtablewidgetitem774->setText(QCoreApplication::translate("InputRZForm", "3", nullptr));
        QTableWidgetItem *___qtablewidgetitem775 = PCUTTable->verticalHeaderItem(0);
        ___qtablewidgetitem775->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem776 = PCUTTable->verticalHeaderItem(1);
        ___qtablewidgetitem776->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem777 = PCUTTable->verticalHeaderItem(2);
        ___qtablewidgetitem777->setText(QCoreApplication::translate("InputRZForm", "3", nullptr));
        QTableWidgetItem *___qtablewidgetitem778 = PCUTTable->verticalHeaderItem(3);
        ___qtablewidgetitem778->setText(QCoreApplication::translate("InputRZForm", "4", nullptr));
        QTableWidgetItem *___qtablewidgetitem779 = PCUTTable->verticalHeaderItem(4);
        ___qtablewidgetitem779->setText(QCoreApplication::translate("InputRZForm", "5", nullptr));
        QTableWidgetItem *___qtablewidgetitem780 = PCUTTable->verticalHeaderItem(5);
        ___qtablewidgetitem780->setText(QCoreApplication::translate("InputRZForm", "6", nullptr));
        QTableWidgetItem *___qtablewidgetitem781 = PCUTTable->verticalHeaderItem(6);
        ___qtablewidgetitem781->setText(QCoreApplication::translate("InputRZForm", "7", nullptr));
        QTableWidgetItem *___qtablewidgetitem782 = PCUTTable->verticalHeaderItem(7);
        ___qtablewidgetitem782->setText(QCoreApplication::translate("InputRZForm", "8", nullptr));
        QTableWidgetItem *___qtablewidgetitem783 = PCUTTable->verticalHeaderItem(8);
        ___qtablewidgetitem783->setText(QCoreApplication::translate("InputRZForm", "9", nullptr));
        QTableWidgetItem *___qtablewidgetitem784 = PCUTTable->verticalHeaderItem(9);
        ___qtablewidgetitem784->setText(QCoreApplication::translate("InputRZForm", "10", nullptr));
        QTableWidgetItem *___qtablewidgetitem785 = PCUTTable->verticalHeaderItem(10);
        ___qtablewidgetitem785->setText(QCoreApplication::translate("InputRZForm", "11", nullptr));
        QTableWidgetItem *___qtablewidgetitem786 = PCUTTable->verticalHeaderItem(11);
        ___qtablewidgetitem786->setText(QCoreApplication::translate("InputRZForm", "12", nullptr));
        QTableWidgetItem *___qtablewidgetitem787 = PCUTTable->verticalHeaderItem(12);
        ___qtablewidgetitem787->setText(QCoreApplication::translate("InputRZForm", "13", nullptr));
        QTableWidgetItem *___qtablewidgetitem788 = PCUTTable->verticalHeaderItem(13);
        ___qtablewidgetitem788->setText(QCoreApplication::translate("InputRZForm", "14", nullptr));
        QTableWidgetItem *___qtablewidgetitem789 = PCUTTable->verticalHeaderItem(14);
        ___qtablewidgetitem789->setText(QCoreApplication::translate("InputRZForm", "15", nullptr));
        QTableWidgetItem *___qtablewidgetitem790 = PCUTTable->verticalHeaderItem(15);
        ___qtablewidgetitem790->setText(QCoreApplication::translate("InputRZForm", "16", nullptr));
        QTableWidgetItem *___qtablewidgetitem791 = PCUTTable->verticalHeaderItem(16);
        ___qtablewidgetitem791->setText(QCoreApplication::translate("InputRZForm", "17", nullptr));
        QTableWidgetItem *___qtablewidgetitem792 = PCUTTable->verticalHeaderItem(17);
        ___qtablewidgetitem792->setText(QCoreApplication::translate("InputRZForm", "18", nullptr));
        QTableWidgetItem *___qtablewidgetitem793 = PCUTTable->verticalHeaderItem(18);
        ___qtablewidgetitem793->setText(QCoreApplication::translate("InputRZForm", "19", nullptr));
        QTableWidgetItem *___qtablewidgetitem794 = PCUTTable->verticalHeaderItem(19);
        ___qtablewidgetitem794->setText(QCoreApplication::translate("InputRZForm", "20", nullptr));
        ECUTGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "ECUT by regions", nullptr));
        QTableWidgetItem *___qtablewidgetitem795 = ECUTTable->horizontalHeaderItem(0);
        ___qtablewidgetitem795->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem796 = ECUTTable->horizontalHeaderItem(1);
        ___qtablewidgetitem796->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem797 = ECUTTable->horizontalHeaderItem(2);
        ___qtablewidgetitem797->setText(QCoreApplication::translate("InputRZForm", "3", nullptr));
        QTableWidgetItem *___qtablewidgetitem798 = ECUTTable->verticalHeaderItem(0);
        ___qtablewidgetitem798->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem799 = ECUTTable->verticalHeaderItem(1);
        ___qtablewidgetitem799->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem800 = ECUTTable->verticalHeaderItem(2);
        ___qtablewidgetitem800->setText(QCoreApplication::translate("InputRZForm", "3", nullptr));
        QTableWidgetItem *___qtablewidgetitem801 = ECUTTable->verticalHeaderItem(3);
        ___qtablewidgetitem801->setText(QCoreApplication::translate("InputRZForm", "4", nullptr));
        QTableWidgetItem *___qtablewidgetitem802 = ECUTTable->verticalHeaderItem(4);
        ___qtablewidgetitem802->setText(QCoreApplication::translate("InputRZForm", "5", nullptr));
        QTableWidgetItem *___qtablewidgetitem803 = ECUTTable->verticalHeaderItem(5);
        ___qtablewidgetitem803->setText(QCoreApplication::translate("InputRZForm", "6", nullptr));
        QTableWidgetItem *___qtablewidgetitem804 = ECUTTable->verticalHeaderItem(6);
        ___qtablewidgetitem804->setText(QCoreApplication::translate("InputRZForm", "7", nullptr));
        QTableWidgetItem *___qtablewidgetitem805 = ECUTTable->verticalHeaderItem(7);
        ___qtablewidgetitem805->setText(QCoreApplication::translate("InputRZForm", "8", nullptr));
        QTableWidgetItem *___qtablewidgetitem806 = ECUTTable->verticalHeaderItem(8);
        ___qtablewidgetitem806->setText(QCoreApplication::translate("InputRZForm", "9", nullptr));
        QTableWidgetItem *___qtablewidgetitem807 = ECUTTable->verticalHeaderItem(9);
        ___qtablewidgetitem807->setText(QCoreApplication::translate("InputRZForm", "10", nullptr));
        QTableWidgetItem *___qtablewidgetitem808 = ECUTTable->verticalHeaderItem(10);
        ___qtablewidgetitem808->setText(QCoreApplication::translate("InputRZForm", "11", nullptr));
        QTableWidgetItem *___qtablewidgetitem809 = ECUTTable->verticalHeaderItem(11);
        ___qtablewidgetitem809->setText(QCoreApplication::translate("InputRZForm", "12", nullptr));
        QTableWidgetItem *___qtablewidgetitem810 = ECUTTable->verticalHeaderItem(12);
        ___qtablewidgetitem810->setText(QCoreApplication::translate("InputRZForm", "13", nullptr));
        QTableWidgetItem *___qtablewidgetitem811 = ECUTTable->verticalHeaderItem(13);
        ___qtablewidgetitem811->setText(QCoreApplication::translate("InputRZForm", "14", nullptr));
        QTableWidgetItem *___qtablewidgetitem812 = ECUTTable->verticalHeaderItem(14);
        ___qtablewidgetitem812->setText(QCoreApplication::translate("InputRZForm", "15", nullptr));
        QTableWidgetItem *___qtablewidgetitem813 = ECUTTable->verticalHeaderItem(15);
        ___qtablewidgetitem813->setText(QCoreApplication::translate("InputRZForm", "16", nullptr));
        QTableWidgetItem *___qtablewidgetitem814 = ECUTTable->verticalHeaderItem(16);
        ___qtablewidgetitem814->setText(QCoreApplication::translate("InputRZForm", "17", nullptr));
        QTableWidgetItem *___qtablewidgetitem815 = ECUTTable->verticalHeaderItem(17);
        ___qtablewidgetitem815->setText(QCoreApplication::translate("InputRZForm", "18", nullptr));
        QTableWidgetItem *___qtablewidgetitem816 = ECUTTable->verticalHeaderItem(18);
        ___qtablewidgetitem816->setText(QCoreApplication::translate("InputRZForm", "19", nullptr));
        QTableWidgetItem *___qtablewidgetitem817 = ECUTTable->verticalHeaderItem(19);
        ___qtablewidgetitem817->setText(QCoreApplication::translate("InputRZForm", "20", nullptr));
        SMAXGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "SMAX by regions", nullptr));
        QTableWidgetItem *___qtablewidgetitem818 = SMAXTable->horizontalHeaderItem(0);
        ___qtablewidgetitem818->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem819 = SMAXTable->horizontalHeaderItem(1);
        ___qtablewidgetitem819->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem820 = SMAXTable->horizontalHeaderItem(2);
        ___qtablewidgetitem820->setText(QCoreApplication::translate("InputRZForm", "3", nullptr));
        QTableWidgetItem *___qtablewidgetitem821 = SMAXTable->verticalHeaderItem(0);
        ___qtablewidgetitem821->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem822 = SMAXTable->verticalHeaderItem(1);
        ___qtablewidgetitem822->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem823 = SMAXTable->verticalHeaderItem(2);
        ___qtablewidgetitem823->setText(QCoreApplication::translate("InputRZForm", "3", nullptr));
        QTableWidgetItem *___qtablewidgetitem824 = SMAXTable->verticalHeaderItem(3);
        ___qtablewidgetitem824->setText(QCoreApplication::translate("InputRZForm", "4", nullptr));
        QTableWidgetItem *___qtablewidgetitem825 = SMAXTable->verticalHeaderItem(4);
        ___qtablewidgetitem825->setText(QCoreApplication::translate("InputRZForm", "5", nullptr));
        QTableWidgetItem *___qtablewidgetitem826 = SMAXTable->verticalHeaderItem(5);
        ___qtablewidgetitem826->setText(QCoreApplication::translate("InputRZForm", "6", nullptr));
        QTableWidgetItem *___qtablewidgetitem827 = SMAXTable->verticalHeaderItem(6);
        ___qtablewidgetitem827->setText(QCoreApplication::translate("InputRZForm", "7", nullptr));
        QTableWidgetItem *___qtablewidgetitem828 = SMAXTable->verticalHeaderItem(7);
        ___qtablewidgetitem828->setText(QCoreApplication::translate("InputRZForm", "8", nullptr));
        QTableWidgetItem *___qtablewidgetitem829 = SMAXTable->verticalHeaderItem(8);
        ___qtablewidgetitem829->setText(QCoreApplication::translate("InputRZForm", "9", nullptr));
        QTableWidgetItem *___qtablewidgetitem830 = SMAXTable->verticalHeaderItem(9);
        ___qtablewidgetitem830->setText(QCoreApplication::translate("InputRZForm", "10", nullptr));
        QTableWidgetItem *___qtablewidgetitem831 = SMAXTable->verticalHeaderItem(10);
        ___qtablewidgetitem831->setText(QCoreApplication::translate("InputRZForm", "11", nullptr));
        QTableWidgetItem *___qtablewidgetitem832 = SMAXTable->verticalHeaderItem(11);
        ___qtablewidgetitem832->setText(QCoreApplication::translate("InputRZForm", "12", nullptr));
        QTableWidgetItem *___qtablewidgetitem833 = SMAXTable->verticalHeaderItem(12);
        ___qtablewidgetitem833->setText(QCoreApplication::translate("InputRZForm", "13", nullptr));
        QTableWidgetItem *___qtablewidgetitem834 = SMAXTable->verticalHeaderItem(13);
        ___qtablewidgetitem834->setText(QCoreApplication::translate("InputRZForm", "14", nullptr));
        QTableWidgetItem *___qtablewidgetitem835 = SMAXTable->verticalHeaderItem(14);
        ___qtablewidgetitem835->setText(QCoreApplication::translate("InputRZForm", "15", nullptr));
        QTableWidgetItem *___qtablewidgetitem836 = SMAXTable->verticalHeaderItem(15);
        ___qtablewidgetitem836->setText(QCoreApplication::translate("InputRZForm", "16", nullptr));
        QTableWidgetItem *___qtablewidgetitem837 = SMAXTable->verticalHeaderItem(16);
        ___qtablewidgetitem837->setText(QCoreApplication::translate("InputRZForm", "17", nullptr));
        QTableWidgetItem *___qtablewidgetitem838 = SMAXTable->verticalHeaderItem(17);
        ___qtablewidgetitem838->setText(QCoreApplication::translate("InputRZForm", "18", nullptr));
        QTableWidgetItem *___qtablewidgetitem839 = SMAXTable->verticalHeaderItem(18);
        ___qtablewidgetitem839->setText(QCoreApplication::translate("InputRZForm", "19", nullptr));
        QTableWidgetItem *___qtablewidgetitem840 = SMAXTable->verticalHeaderItem(19);
        ___qtablewidgetitem840->setText(QCoreApplication::translate("InputRZForm", "20", nullptr));
#if QT_CONFIG(tooltip)
        BoundComptonGroupBox->setToolTip(QString());
#endif // QT_CONFIG(tooltip)
        BoundComptonGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Bound Compton scattering", nullptr));
        QTableWidgetItem *___qtablewidgetitem841 = BoundComptonTable->horizontalHeaderItem(0);
        ___qtablewidgetitem841->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem842 = BoundComptonTable->horizontalHeaderItem(1);
        ___qtablewidgetitem842->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem843 = BoundComptonTable->verticalHeaderItem(0);
        ___qtablewidgetitem843->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem844 = BoundComptonTable->verticalHeaderItem(1);
        ___qtablewidgetitem844->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem845 = BoundComptonTable->verticalHeaderItem(2);
        ___qtablewidgetitem845->setText(QCoreApplication::translate("InputRZForm", "3", nullptr));
        QTableWidgetItem *___qtablewidgetitem846 = BoundComptonTable->verticalHeaderItem(3);
        ___qtablewidgetitem846->setText(QCoreApplication::translate("InputRZForm", "4", nullptr));
        QTableWidgetItem *___qtablewidgetitem847 = BoundComptonTable->verticalHeaderItem(4);
        ___qtablewidgetitem847->setText(QCoreApplication::translate("InputRZForm", "5", nullptr));
        QTableWidgetItem *___qtablewidgetitem848 = BoundComptonTable->verticalHeaderItem(5);
        ___qtablewidgetitem848->setText(QCoreApplication::translate("InputRZForm", "6", nullptr));
        QTableWidgetItem *___qtablewidgetitem849 = BoundComptonTable->verticalHeaderItem(6);
        ___qtablewidgetitem849->setText(QCoreApplication::translate("InputRZForm", "7", nullptr));
        QTableWidgetItem *___qtablewidgetitem850 = BoundComptonTable->verticalHeaderItem(7);
        ___qtablewidgetitem850->setText(QCoreApplication::translate("InputRZForm", "8", nullptr));
        QTableWidgetItem *___qtablewidgetitem851 = BoundComptonTable->verticalHeaderItem(8);
        ___qtablewidgetitem851->setText(QCoreApplication::translate("InputRZForm", "9", nullptr));
        QTableWidgetItem *___qtablewidgetitem852 = BoundComptonTable->verticalHeaderItem(9);
        ___qtablewidgetitem852->setText(QCoreApplication::translate("InputRZForm", "10", nullptr));
        QTableWidgetItem *___qtablewidgetitem853 = BoundComptonTable->verticalHeaderItem(10);
        ___qtablewidgetitem853->setText(QCoreApplication::translate("InputRZForm", "11", nullptr));
        QTableWidgetItem *___qtablewidgetitem854 = BoundComptonTable->verticalHeaderItem(11);
        ___qtablewidgetitem854->setText(QCoreApplication::translate("InputRZForm", "12", nullptr));
        QTableWidgetItem *___qtablewidgetitem855 = BoundComptonTable->verticalHeaderItem(12);
        ___qtablewidgetitem855->setText(QCoreApplication::translate("InputRZForm", "13", nullptr));
        QTableWidgetItem *___qtablewidgetitem856 = BoundComptonTable->verticalHeaderItem(13);
        ___qtablewidgetitem856->setText(QCoreApplication::translate("InputRZForm", "14", nullptr));
        QTableWidgetItem *___qtablewidgetitem857 = BoundComptonTable->verticalHeaderItem(14);
        ___qtablewidgetitem857->setText(QCoreApplication::translate("InputRZForm", "15", nullptr));
        QTableWidgetItem *___qtablewidgetitem858 = BoundComptonTable->verticalHeaderItem(15);
        ___qtablewidgetitem858->setText(QCoreApplication::translate("InputRZForm", "16", nullptr));
        QTableWidgetItem *___qtablewidgetitem859 = BoundComptonTable->verticalHeaderItem(16);
        ___qtablewidgetitem859->setText(QCoreApplication::translate("InputRZForm", "17", nullptr));
        QTableWidgetItem *___qtablewidgetitem860 = BoundComptonTable->verticalHeaderItem(17);
        ___qtablewidgetitem860->setText(QCoreApplication::translate("InputRZForm", "18", nullptr));
        QTableWidgetItem *___qtablewidgetitem861 = BoundComptonTable->verticalHeaderItem(18);
        ___qtablewidgetitem861->setText(QCoreApplication::translate("InputRZForm", "19", nullptr));
        QTableWidgetItem *___qtablewidgetitem862 = BoundComptonTable->verticalHeaderItem(19);
        ___qtablewidgetitem862->setText(QCoreApplication::translate("InputRZForm", "20", nullptr));
#if QT_CONFIG(tooltip)
        RelaxationsGroupBox->setToolTip(QString());
#endif // QT_CONFIG(tooltip)
        RelaxationsGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Atomic Relaxations", nullptr));
        QTableWidgetItem *___qtablewidgetitem863 = RelaxationsTable->horizontalHeaderItem(0);
        ___qtablewidgetitem863->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem864 = RelaxationsTable->horizontalHeaderItem(1);
        ___qtablewidgetitem864->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem865 = RelaxationsTable->verticalHeaderItem(0);
        ___qtablewidgetitem865->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem866 = RelaxationsTable->verticalHeaderItem(1);
        ___qtablewidgetitem866->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem867 = RelaxationsTable->verticalHeaderItem(2);
        ___qtablewidgetitem867->setText(QCoreApplication::translate("InputRZForm", "3", nullptr));
        QTableWidgetItem *___qtablewidgetitem868 = RelaxationsTable->verticalHeaderItem(3);
        ___qtablewidgetitem868->setText(QCoreApplication::translate("InputRZForm", "4", nullptr));
        QTableWidgetItem *___qtablewidgetitem869 = RelaxationsTable->verticalHeaderItem(4);
        ___qtablewidgetitem869->setText(QCoreApplication::translate("InputRZForm", "5", nullptr));
        QTableWidgetItem *___qtablewidgetitem870 = RelaxationsTable->verticalHeaderItem(5);
        ___qtablewidgetitem870->setText(QCoreApplication::translate("InputRZForm", "6", nullptr));
        QTableWidgetItem *___qtablewidgetitem871 = RelaxationsTable->verticalHeaderItem(6);
        ___qtablewidgetitem871->setText(QCoreApplication::translate("InputRZForm", "7", nullptr));
        QTableWidgetItem *___qtablewidgetitem872 = RelaxationsTable->verticalHeaderItem(7);
        ___qtablewidgetitem872->setText(QCoreApplication::translate("InputRZForm", "8", nullptr));
        QTableWidgetItem *___qtablewidgetitem873 = RelaxationsTable->verticalHeaderItem(8);
        ___qtablewidgetitem873->setText(QCoreApplication::translate("InputRZForm", "9", nullptr));
        QTableWidgetItem *___qtablewidgetitem874 = RelaxationsTable->verticalHeaderItem(9);
        ___qtablewidgetitem874->setText(QCoreApplication::translate("InputRZForm", "10", nullptr));
        QTableWidgetItem *___qtablewidgetitem875 = RelaxationsTable->verticalHeaderItem(10);
        ___qtablewidgetitem875->setText(QCoreApplication::translate("InputRZForm", "11", nullptr));
        QTableWidgetItem *___qtablewidgetitem876 = RelaxationsTable->verticalHeaderItem(11);
        ___qtablewidgetitem876->setText(QCoreApplication::translate("InputRZForm", "12", nullptr));
        QTableWidgetItem *___qtablewidgetitem877 = RelaxationsTable->verticalHeaderItem(12);
        ___qtablewidgetitem877->setText(QCoreApplication::translate("InputRZForm", "13", nullptr));
        QTableWidgetItem *___qtablewidgetitem878 = RelaxationsTable->verticalHeaderItem(13);
        ___qtablewidgetitem878->setText(QCoreApplication::translate("InputRZForm", "14", nullptr));
        QTableWidgetItem *___qtablewidgetitem879 = RelaxationsTable->verticalHeaderItem(14);
        ___qtablewidgetitem879->setText(QCoreApplication::translate("InputRZForm", "15", nullptr));
        QTableWidgetItem *___qtablewidgetitem880 = RelaxationsTable->verticalHeaderItem(15);
        ___qtablewidgetitem880->setText(QCoreApplication::translate("InputRZForm", "16", nullptr));
        QTableWidgetItem *___qtablewidgetitem881 = RelaxationsTable->verticalHeaderItem(16);
        ___qtablewidgetitem881->setText(QCoreApplication::translate("InputRZForm", "17", nullptr));
        QTableWidgetItem *___qtablewidgetitem882 = RelaxationsTable->verticalHeaderItem(17);
        ___qtablewidgetitem882->setText(QCoreApplication::translate("InputRZForm", "18", nullptr));
        QTableWidgetItem *___qtablewidgetitem883 = RelaxationsTable->verticalHeaderItem(18);
        ___qtablewidgetitem883->setText(QCoreApplication::translate("InputRZForm", "19", nullptr));
        QTableWidgetItem *___qtablewidgetitem884 = RelaxationsTable->verticalHeaderItem(19);
        ___qtablewidgetitem884->setText(QCoreApplication::translate("InputRZForm", "20", nullptr));
#if QT_CONFIG(tooltip)
        PEAngSamGroupBox->setToolTip(QString());
#endif // QT_CONFIG(tooltip)
        PEAngSamGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "PE ang. sampling", nullptr));
        QTableWidgetItem *___qtablewidgetitem885 = PEAngSamplingTable->horizontalHeaderItem(0);
        ___qtablewidgetitem885->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem886 = PEAngSamplingTable->horizontalHeaderItem(1);
        ___qtablewidgetitem886->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem887 = PEAngSamplingTable->verticalHeaderItem(0);
        ___qtablewidgetitem887->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem888 = PEAngSamplingTable->verticalHeaderItem(1);
        ___qtablewidgetitem888->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem889 = PEAngSamplingTable->verticalHeaderItem(2);
        ___qtablewidgetitem889->setText(QCoreApplication::translate("InputRZForm", "3", nullptr));
        QTableWidgetItem *___qtablewidgetitem890 = PEAngSamplingTable->verticalHeaderItem(3);
        ___qtablewidgetitem890->setText(QCoreApplication::translate("InputRZForm", "4", nullptr));
        QTableWidgetItem *___qtablewidgetitem891 = PEAngSamplingTable->verticalHeaderItem(4);
        ___qtablewidgetitem891->setText(QCoreApplication::translate("InputRZForm", "5", nullptr));
        QTableWidgetItem *___qtablewidgetitem892 = PEAngSamplingTable->verticalHeaderItem(5);
        ___qtablewidgetitem892->setText(QCoreApplication::translate("InputRZForm", "6", nullptr));
        QTableWidgetItem *___qtablewidgetitem893 = PEAngSamplingTable->verticalHeaderItem(6);
        ___qtablewidgetitem893->setText(QCoreApplication::translate("InputRZForm", "7", nullptr));
        QTableWidgetItem *___qtablewidgetitem894 = PEAngSamplingTable->verticalHeaderItem(7);
        ___qtablewidgetitem894->setText(QCoreApplication::translate("InputRZForm", "8", nullptr));
        QTableWidgetItem *___qtablewidgetitem895 = PEAngSamplingTable->verticalHeaderItem(8);
        ___qtablewidgetitem895->setText(QCoreApplication::translate("InputRZForm", "9", nullptr));
        QTableWidgetItem *___qtablewidgetitem896 = PEAngSamplingTable->verticalHeaderItem(9);
        ___qtablewidgetitem896->setText(QCoreApplication::translate("InputRZForm", "10", nullptr));
        QTableWidgetItem *___qtablewidgetitem897 = PEAngSamplingTable->verticalHeaderItem(10);
        ___qtablewidgetitem897->setText(QCoreApplication::translate("InputRZForm", "11", nullptr));
        QTableWidgetItem *___qtablewidgetitem898 = PEAngSamplingTable->verticalHeaderItem(11);
        ___qtablewidgetitem898->setText(QCoreApplication::translate("InputRZForm", "12", nullptr));
        QTableWidgetItem *___qtablewidgetitem899 = PEAngSamplingTable->verticalHeaderItem(12);
        ___qtablewidgetitem899->setText(QCoreApplication::translate("InputRZForm", "13", nullptr));
        QTableWidgetItem *___qtablewidgetitem900 = PEAngSamplingTable->verticalHeaderItem(13);
        ___qtablewidgetitem900->setText(QCoreApplication::translate("InputRZForm", "14", nullptr));
        QTableWidgetItem *___qtablewidgetitem901 = PEAngSamplingTable->verticalHeaderItem(14);
        ___qtablewidgetitem901->setText(QCoreApplication::translate("InputRZForm", "15", nullptr));
        QTableWidgetItem *___qtablewidgetitem902 = PEAngSamplingTable->verticalHeaderItem(15);
        ___qtablewidgetitem902->setText(QCoreApplication::translate("InputRZForm", "16", nullptr));
        QTableWidgetItem *___qtablewidgetitem903 = PEAngSamplingTable->verticalHeaderItem(16);
        ___qtablewidgetitem903->setText(QCoreApplication::translate("InputRZForm", "17", nullptr));
        QTableWidgetItem *___qtablewidgetitem904 = PEAngSamplingTable->verticalHeaderItem(17);
        ___qtablewidgetitem904->setText(QCoreApplication::translate("InputRZForm", "18", nullptr));
        QTableWidgetItem *___qtablewidgetitem905 = PEAngSamplingTable->verticalHeaderItem(18);
        ___qtablewidgetitem905->setText(QCoreApplication::translate("InputRZForm", "19", nullptr));
        QTableWidgetItem *___qtablewidgetitem906 = PEAngSamplingTable->verticalHeaderItem(19);
        ___qtablewidgetitem906->setText(QCoreApplication::translate("InputRZForm", "20", nullptr));
#if QT_CONFIG(tooltip)
        RayleighGroupBox->setToolTip(QString());
#endif // QT_CONFIG(tooltip)
        RayleighGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Rayleigh scattering", nullptr));
        QTableWidgetItem *___qtablewidgetitem907 = RayleighTable->horizontalHeaderItem(0);
        ___qtablewidgetitem907->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem908 = RayleighTable->horizontalHeaderItem(1);
        ___qtablewidgetitem908->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem909 = RayleighTable->verticalHeaderItem(0);
        ___qtablewidgetitem909->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem910 = RayleighTable->verticalHeaderItem(1);
        ___qtablewidgetitem910->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem911 = RayleighTable->verticalHeaderItem(2);
        ___qtablewidgetitem911->setText(QCoreApplication::translate("InputRZForm", "3", nullptr));
        QTableWidgetItem *___qtablewidgetitem912 = RayleighTable->verticalHeaderItem(3);
        ___qtablewidgetitem912->setText(QCoreApplication::translate("InputRZForm", "4", nullptr));
        QTableWidgetItem *___qtablewidgetitem913 = RayleighTable->verticalHeaderItem(4);
        ___qtablewidgetitem913->setText(QCoreApplication::translate("InputRZForm", "5", nullptr));
        QTableWidgetItem *___qtablewidgetitem914 = RayleighTable->verticalHeaderItem(5);
        ___qtablewidgetitem914->setText(QCoreApplication::translate("InputRZForm", "6", nullptr));
        QTableWidgetItem *___qtablewidgetitem915 = RayleighTable->verticalHeaderItem(6);
        ___qtablewidgetitem915->setText(QCoreApplication::translate("InputRZForm", "7", nullptr));
        QTableWidgetItem *___qtablewidgetitem916 = RayleighTable->verticalHeaderItem(7);
        ___qtablewidgetitem916->setText(QCoreApplication::translate("InputRZForm", "8", nullptr));
        QTableWidgetItem *___qtablewidgetitem917 = RayleighTable->verticalHeaderItem(8);
        ___qtablewidgetitem917->setText(QCoreApplication::translate("InputRZForm", "9", nullptr));
        QTableWidgetItem *___qtablewidgetitem918 = RayleighTable->verticalHeaderItem(9);
        ___qtablewidgetitem918->setText(QCoreApplication::translate("InputRZForm", "10", nullptr));
        QTableWidgetItem *___qtablewidgetitem919 = RayleighTable->verticalHeaderItem(10);
        ___qtablewidgetitem919->setText(QCoreApplication::translate("InputRZForm", "11", nullptr));
        QTableWidgetItem *___qtablewidgetitem920 = RayleighTable->verticalHeaderItem(11);
        ___qtablewidgetitem920->setText(QCoreApplication::translate("InputRZForm", "12", nullptr));
        QTableWidgetItem *___qtablewidgetitem921 = RayleighTable->verticalHeaderItem(12);
        ___qtablewidgetitem921->setText(QCoreApplication::translate("InputRZForm", "13", nullptr));
        QTableWidgetItem *___qtablewidgetitem922 = RayleighTable->verticalHeaderItem(13);
        ___qtablewidgetitem922->setText(QCoreApplication::translate("InputRZForm", "14", nullptr));
        QTableWidgetItem *___qtablewidgetitem923 = RayleighTable->verticalHeaderItem(14);
        ___qtablewidgetitem923->setText(QCoreApplication::translate("InputRZForm", "15", nullptr));
        QTableWidgetItem *___qtablewidgetitem924 = RayleighTable->verticalHeaderItem(15);
        ___qtablewidgetitem924->setText(QCoreApplication::translate("InputRZForm", "16", nullptr));
        QTableWidgetItem *___qtablewidgetitem925 = RayleighTable->verticalHeaderItem(16);
        ___qtablewidgetitem925->setText(QCoreApplication::translate("InputRZForm", "17", nullptr));
        QTableWidgetItem *___qtablewidgetitem926 = RayleighTable->verticalHeaderItem(17);
        ___qtablewidgetitem926->setText(QCoreApplication::translate("InputRZForm", "18", nullptr));
        QTableWidgetItem *___qtablewidgetitem927 = RayleighTable->verticalHeaderItem(18);
        ___qtablewidgetitem927->setText(QCoreApplication::translate("InputRZForm", "19", nullptr));
        QTableWidgetItem *___qtablewidgetitem928 = RayleighTable->verticalHeaderItem(19);
        ___qtablewidgetitem928->setText(QCoreApplication::translate("InputRZForm", "20", nullptr));
        TabWidgetRZ->setTabText(TabWidgetRZ->indexOf(MCTPRegTab), QCoreApplication::translate("InputRZForm", "Transport Parameters by Region", nullptr));
        plotCheckBox->setText(QCoreApplication::translate("InputRZForm", "Plotting", nullptr));
        PlotGroupBox->setTitle(QString());
        LinePrnOutCheckBox->setText(QCoreApplication::translate("InputRZForm", "Line printer output", nullptr));
        ExtPlotOutCheckBox->setText(QCoreApplication::translate("InputRZForm", "External plotter output", nullptr));
        ExternalPlotTypeGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "External plot type", nullptr));
        ExternalPlotTypeComboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "Point", nullptr));
        ExternalPlotTypeComboBox->setItemText(1, QCoreApplication::translate("InputRZForm", "Histogram", nullptr));
        ExternalPlotTypeComboBox->setItemText(2, QCoreApplication::translate("InputRZForm", "Both", nullptr));

        PlotRegionsGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Define plot regions", nullptr));
        QTableWidgetItem *___qtablewidgetitem929 = PlotRegionsTable->horizontalHeaderItem(0);
        ___qtablewidgetitem929->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem930 = PlotRegionsTable->horizontalHeaderItem(1);
        ___qtablewidgetitem930->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem931 = PlotRegionsTable->verticalHeaderItem(0);
        ___qtablewidgetitem931->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem932 = PlotRegionsTable->verticalHeaderItem(1);
        ___qtablewidgetitem932->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem933 = PlotRegionsTable->verticalHeaderItem(2);
        ___qtablewidgetitem933->setText(QCoreApplication::translate("InputRZForm", "3", nullptr));
        QTableWidgetItem *___qtablewidgetitem934 = PlotRegionsTable->verticalHeaderItem(3);
        ___qtablewidgetitem934->setText(QCoreApplication::translate("InputRZForm", "4", nullptr));
        QTableWidgetItem *___qtablewidgetitem935 = PlotRegionsTable->verticalHeaderItem(4);
        ___qtablewidgetitem935->setText(QCoreApplication::translate("InputRZForm", "5", nullptr));
        QTableWidgetItem *___qtablewidgetitem936 = PlotRegionsTable->verticalHeaderItem(5);
        ___qtablewidgetitem936->setText(QCoreApplication::translate("InputRZForm", "6", nullptr));
        QTableWidgetItem *___qtablewidgetitem937 = PlotRegionsTable->verticalHeaderItem(6);
        ___qtablewidgetitem937->setText(QCoreApplication::translate("InputRZForm", "7", nullptr));
        QTableWidgetItem *___qtablewidgetitem938 = PlotRegionsTable->verticalHeaderItem(7);
        ___qtablewidgetitem938->setText(QCoreApplication::translate("InputRZForm", "8", nullptr));
        QTableWidgetItem *___qtablewidgetitem939 = PlotRegionsTable->verticalHeaderItem(8);
        ___qtablewidgetitem939->setText(QCoreApplication::translate("InputRZForm", "9", nullptr));
        QTableWidgetItem *___qtablewidgetitem940 = PlotRegionsTable->verticalHeaderItem(9);
        ___qtablewidgetitem940->setText(QCoreApplication::translate("InputRZForm", "10", nullptr));
        QTableWidgetItem *___qtablewidgetitem941 = PlotRegionsTable->verticalHeaderItem(10);
        ___qtablewidgetitem941->setText(QCoreApplication::translate("InputRZForm", "11", nullptr));
        QTableWidgetItem *___qtablewidgetitem942 = PlotRegionsTable->verticalHeaderItem(11);
        ___qtablewidgetitem942->setText(QCoreApplication::translate("InputRZForm", "12", nullptr));
        QTableWidgetItem *___qtablewidgetitem943 = PlotRegionsTable->verticalHeaderItem(12);
        ___qtablewidgetitem943->setText(QCoreApplication::translate("InputRZForm", "13", nullptr));
        QTableWidgetItem *___qtablewidgetitem944 = PlotRegionsTable->verticalHeaderItem(13);
        ___qtablewidgetitem944->setText(QCoreApplication::translate("InputRZForm", "14", nullptr));
        QTableWidgetItem *___qtablewidgetitem945 = PlotRegionsTable->verticalHeaderItem(14);
        ___qtablewidgetitem945->setText(QCoreApplication::translate("InputRZForm", "15", nullptr));
        QTableWidgetItem *___qtablewidgetitem946 = PlotRegionsTable->verticalHeaderItem(15);
        ___qtablewidgetitem946->setText(QCoreApplication::translate("InputRZForm", "16", nullptr));
        QTableWidgetItem *___qtablewidgetitem947 = PlotRegionsTable->verticalHeaderItem(16);
        ___qtablewidgetitem947->setText(QCoreApplication::translate("InputRZForm", "17", nullptr));
        QTableWidgetItem *___qtablewidgetitem948 = PlotRegionsTable->verticalHeaderItem(17);
        ___qtablewidgetitem948->setText(QCoreApplication::translate("InputRZForm", "18", nullptr));
        QTableWidgetItem *___qtablewidgetitem949 = PlotRegionsTable->verticalHeaderItem(18);
        ___qtablewidgetitem949->setText(QCoreApplication::translate("InputRZForm", "19", nullptr));
        QTableWidgetItem *___qtablewidgetitem950 = PlotRegionsTable->verticalHeaderItem(19);
        ___qtablewidgetitem950->setText(QCoreApplication::translate("InputRZForm", "20", nullptr));
        QTableWidgetItem *___qtablewidgetitem951 = PlotRegionsTable->verticalHeaderItem(20);
        ___qtablewidgetitem951->setText(QCoreApplication::translate("InputRZForm", "21", nullptr));
        QTableWidgetItem *___qtablewidgetitem952 = PlotRegionsTable->verticalHeaderItem(21);
        ___qtablewidgetitem952->setText(QCoreApplication::translate("InputRZForm", "22", nullptr));
        QTableWidgetItem *___qtablewidgetitem953 = PlotRegionsTable->verticalHeaderItem(22);
        ___qtablewidgetitem953->setText(QCoreApplication::translate("InputRZForm", "23", nullptr));
        QTableWidgetItem *___qtablewidgetitem954 = PlotRegionsTable->verticalHeaderItem(23);
        ___qtablewidgetitem954->setText(QCoreApplication::translate("InputRZForm", "24", nullptr));
        QTableWidgetItem *___qtablewidgetitem955 = PlotRegionsTable->verticalHeaderItem(24);
        ___qtablewidgetitem955->setText(QCoreApplication::translate("InputRZForm", "25", nullptr));
        QTableWidgetItem *___qtablewidgetitem956 = PlotRegionsTable->verticalHeaderItem(25);
        ___qtablewidgetitem956->setText(QCoreApplication::translate("InputRZForm", "26", nullptr));
        QTableWidgetItem *___qtablewidgetitem957 = PlotRegionsTable->verticalHeaderItem(26);
        ___qtablewidgetitem957->setText(QCoreApplication::translate("InputRZForm", "27", nullptr));
        QTableWidgetItem *___qtablewidgetitem958 = PlotRegionsTable->verticalHeaderItem(27);
        ___qtablewidgetitem958->setText(QCoreApplication::translate("InputRZForm", "28", nullptr));
        QTableWidgetItem *___qtablewidgetitem959 = PlotRegionsTable->verticalHeaderItem(28);
        ___qtablewidgetitem959->setText(QCoreApplication::translate("InputRZForm", "29", nullptr));
        QTableWidgetItem *___qtablewidgetitem960 = PlotRegionsTable->verticalHeaderItem(29);
        ___qtablewidgetitem960->setText(QCoreApplication::translate("InputRZForm", "30", nullptr));
        QTableWidgetItem *___qtablewidgetitem961 = PlotRegionsTable->verticalHeaderItem(30);
        ___qtablewidgetitem961->setText(QCoreApplication::translate("InputRZForm", "31", nullptr));
        QTableWidgetItem *___qtablewidgetitem962 = PlotRegionsTable->verticalHeaderItem(31);
        ___qtablewidgetitem962->setText(QCoreApplication::translate("InputRZForm", "32", nullptr));
        QTableWidgetItem *___qtablewidgetitem963 = PlotRegionsTable->verticalHeaderItem(32);
        ___qtablewidgetitem963->setText(QCoreApplication::translate("InputRZForm", "33", nullptr));
        QTableWidgetItem *___qtablewidgetitem964 = PlotRegionsTable->verticalHeaderItem(33);
        ___qtablewidgetitem964->setText(QCoreApplication::translate("InputRZForm", "34", nullptr));
        QTableWidgetItem *___qtablewidgetitem965 = PlotRegionsTable->verticalHeaderItem(34);
        ___qtablewidgetitem965->setText(QCoreApplication::translate("InputRZForm", "35", nullptr));
        QTableWidgetItem *___qtablewidgetitem966 = PlotRegionsTable->verticalHeaderItem(35);
        ___qtablewidgetitem966->setText(QCoreApplication::translate("InputRZForm", "36", nullptr));
        QTableWidgetItem *___qtablewidgetitem967 = PlotRegionsTable->verticalHeaderItem(36);
        ___qtablewidgetitem967->setText(QCoreApplication::translate("InputRZForm", "37", nullptr));
        QTableWidgetItem *___qtablewidgetitem968 = PlotRegionsTable->verticalHeaderItem(37);
        ___qtablewidgetitem968->setText(QCoreApplication::translate("InputRZForm", "38", nullptr));
        QTableWidgetItem *___qtablewidgetitem969 = PlotRegionsTable->verticalHeaderItem(38);
        ___qtablewidgetitem969->setText(QCoreApplication::translate("InputRZForm", "39", nullptr));
        QTableWidgetItem *___qtablewidgetitem970 = PlotRegionsTable->verticalHeaderItem(39);
        ___qtablewidgetitem970->setText(QCoreApplication::translate("InputRZForm", "40", nullptr));
        QTableWidgetItem *___qtablewidgetitem971 = PlotRegionsTable->verticalHeaderItem(40);
        ___qtablewidgetitem971->setText(QCoreApplication::translate("InputRZForm", "41", nullptr));
        QTableWidgetItem *___qtablewidgetitem972 = PlotRegionsTable->verticalHeaderItem(41);
        ___qtablewidgetitem972->setText(QCoreApplication::translate("InputRZForm", "42", nullptr));
        QTableWidgetItem *___qtablewidgetitem973 = PlotRegionsTable->verticalHeaderItem(42);
        ___qtablewidgetitem973->setText(QCoreApplication::translate("InputRZForm", "43", nullptr));
        QTableWidgetItem *___qtablewidgetitem974 = PlotRegionsTable->verticalHeaderItem(43);
        ___qtablewidgetitem974->setText(QCoreApplication::translate("InputRZForm", "44", nullptr));
        QTableWidgetItem *___qtablewidgetitem975 = PlotRegionsTable->verticalHeaderItem(44);
        ___qtablewidgetitem975->setText(QCoreApplication::translate("InputRZForm", "45", nullptr));
        QTableWidgetItem *___qtablewidgetitem976 = PlotRegionsTable->verticalHeaderItem(45);
        ___qtablewidgetitem976->setText(QCoreApplication::translate("InputRZForm", "46", nullptr));
        QTableWidgetItem *___qtablewidgetitem977 = PlotRegionsTable->verticalHeaderItem(46);
        ___qtablewidgetitem977->setText(QCoreApplication::translate("InputRZForm", "47", nullptr));
        QTableWidgetItem *___qtablewidgetitem978 = PlotRegionsTable->verticalHeaderItem(47);
        ___qtablewidgetitem978->setText(QCoreApplication::translate("InputRZForm", "48", nullptr));
        QTableWidgetItem *___qtablewidgetitem979 = PlotRegionsTable->verticalHeaderItem(48);
        ___qtablewidgetitem979->setText(QCoreApplication::translate("InputRZForm", "49", nullptr));
        QTableWidgetItem *___qtablewidgetitem980 = PlotRegionsTable->verticalHeaderItem(49);
        ___qtablewidgetitem980->setText(QCoreApplication::translate("InputRZForm", "50", nullptr));
        eminusPlotCheckBox->setText(QCoreApplication::translate("InputRZForm", "Plots for electrons", nullptr));
        eplusPlotCheckBox->setText(QCoreApplication::translate("InputRZForm", "Plots for positrons", nullptr));
        gammaPlotCheckBox->setText(QCoreApplication::translate("InputRZForm", "Plots for photons", nullptr));
        ePlotCheckBox->setText(QCoreApplication::translate("InputRZForm", "Plots for e- and e+", nullptr));
        DrawFluPlotsGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Type of integral fluence to plot", nullptr));
        DrawFluPlotsComboBox->setItemText(0, QCoreApplication::translate("InputRZForm", "Primaries and total", nullptr));
        DrawFluPlotsComboBox->setItemText(1, QCoreApplication::translate("InputRZForm", "Primaries", nullptr));
        DrawFluPlotsComboBox->setItemText(2, QCoreApplication::translate("InputRZForm", "Total", nullptr));

        SpecPlotGroupBox->setTitle(QCoreApplication::translate("InputRZForm", "Spectral plot regions", nullptr));
        QTableWidgetItem *___qtablewidgetitem981 = SpecPlotTable->horizontalHeaderItem(0);
        ___qtablewidgetitem981->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem982 = SpecPlotTable->horizontalHeaderItem(1);
        ___qtablewidgetitem982->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem983 = SpecPlotTable->verticalHeaderItem(0);
        ___qtablewidgetitem983->setText(QCoreApplication::translate("InputRZForm", "1", nullptr));
        QTableWidgetItem *___qtablewidgetitem984 = SpecPlotTable->verticalHeaderItem(1);
        ___qtablewidgetitem984->setText(QCoreApplication::translate("InputRZForm", "2", nullptr));
        QTableWidgetItem *___qtablewidgetitem985 = SpecPlotTable->verticalHeaderItem(2);
        ___qtablewidgetitem985->setText(QCoreApplication::translate("InputRZForm", "3", nullptr));
        QTableWidgetItem *___qtablewidgetitem986 = SpecPlotTable->verticalHeaderItem(3);
        ___qtablewidgetitem986->setText(QCoreApplication::translate("InputRZForm", "4", nullptr));
        QTableWidgetItem *___qtablewidgetitem987 = SpecPlotTable->verticalHeaderItem(4);
        ___qtablewidgetitem987->setText(QCoreApplication::translate("InputRZForm", "5", nullptr));
        QTableWidgetItem *___qtablewidgetitem988 = SpecPlotTable->verticalHeaderItem(5);
        ___qtablewidgetitem988->setText(QCoreApplication::translate("InputRZForm", "6", nullptr));
        QTableWidgetItem *___qtablewidgetitem989 = SpecPlotTable->verticalHeaderItem(6);
        ___qtablewidgetitem989->setText(QCoreApplication::translate("InputRZForm", "7", nullptr));
        QTableWidgetItem *___qtablewidgetitem990 = SpecPlotTable->verticalHeaderItem(7);
        ___qtablewidgetitem990->setText(QCoreApplication::translate("InputRZForm", "8", nullptr));
        QTableWidgetItem *___qtablewidgetitem991 = SpecPlotTable->verticalHeaderItem(8);
        ___qtablewidgetitem991->setText(QCoreApplication::translate("InputRZForm", "9", nullptr));
        QTableWidgetItem *___qtablewidgetitem992 = SpecPlotTable->verticalHeaderItem(9);
        ___qtablewidgetitem992->setText(QCoreApplication::translate("InputRZForm", "10", nullptr));
        QTableWidgetItem *___qtablewidgetitem993 = SpecPlotTable->verticalHeaderItem(10);
        ___qtablewidgetitem993->setText(QCoreApplication::translate("InputRZForm", "11", nullptr));
        QTableWidgetItem *___qtablewidgetitem994 = SpecPlotTable->verticalHeaderItem(11);
        ___qtablewidgetitem994->setText(QCoreApplication::translate("InputRZForm", "12", nullptr));
        QTableWidgetItem *___qtablewidgetitem995 = SpecPlotTable->verticalHeaderItem(12);
        ___qtablewidgetitem995->setText(QCoreApplication::translate("InputRZForm", "13", nullptr));
        QTableWidgetItem *___qtablewidgetitem996 = SpecPlotTable->verticalHeaderItem(13);
        ___qtablewidgetitem996->setText(QCoreApplication::translate("InputRZForm", "14", nullptr));
        QTableWidgetItem *___qtablewidgetitem997 = SpecPlotTable->verticalHeaderItem(14);
        ___qtablewidgetitem997->setText(QCoreApplication::translate("InputRZForm", "15", nullptr));
        QTableWidgetItem *___qtablewidgetitem998 = SpecPlotTable->verticalHeaderItem(15);
        ___qtablewidgetitem998->setText(QCoreApplication::translate("InputRZForm", "16", nullptr));
        QTableWidgetItem *___qtablewidgetitem999 = SpecPlotTable->verticalHeaderItem(16);
        ___qtablewidgetitem999->setText(QCoreApplication::translate("InputRZForm", "17", nullptr));
        QTableWidgetItem *___qtablewidgetitem1000 = SpecPlotTable->verticalHeaderItem(17);
        ___qtablewidgetitem1000->setText(QCoreApplication::translate("InputRZForm", "18", nullptr));
        QTableWidgetItem *___qtablewidgetitem1001 = SpecPlotTable->verticalHeaderItem(18);
        ___qtablewidgetitem1001->setText(QCoreApplication::translate("InputRZForm", "19", nullptr));
        QTableWidgetItem *___qtablewidgetitem1002 = SpecPlotTable->verticalHeaderItem(19);
        ___qtablewidgetitem1002->setText(QCoreApplication::translate("InputRZForm", "20", nullptr));
        QTableWidgetItem *___qtablewidgetitem1003 = SpecPlotTable->verticalHeaderItem(20);
        ___qtablewidgetitem1003->setText(QCoreApplication::translate("InputRZForm", "21", nullptr));
        QTableWidgetItem *___qtablewidgetitem1004 = SpecPlotTable->verticalHeaderItem(21);
        ___qtablewidgetitem1004->setText(QCoreApplication::translate("InputRZForm", "22", nullptr));
        QTableWidgetItem *___qtablewidgetitem1005 = SpecPlotTable->verticalHeaderItem(22);
        ___qtablewidgetitem1005->setText(QCoreApplication::translate("InputRZForm", "23", nullptr));
        QTableWidgetItem *___qtablewidgetitem1006 = SpecPlotTable->verticalHeaderItem(23);
        ___qtablewidgetitem1006->setText(QCoreApplication::translate("InputRZForm", "24", nullptr));
        QTableWidgetItem *___qtablewidgetitem1007 = SpecPlotTable->verticalHeaderItem(24);
        ___qtablewidgetitem1007->setText(QCoreApplication::translate("InputRZForm", "25", nullptr));
        QTableWidgetItem *___qtablewidgetitem1008 = SpecPlotTable->verticalHeaderItem(25);
        ___qtablewidgetitem1008->setText(QCoreApplication::translate("InputRZForm", "26", nullptr));
        QTableWidgetItem *___qtablewidgetitem1009 = SpecPlotTable->verticalHeaderItem(26);
        ___qtablewidgetitem1009->setText(QCoreApplication::translate("InputRZForm", "27", nullptr));
        QTableWidgetItem *___qtablewidgetitem1010 = SpecPlotTable->verticalHeaderItem(27);
        ___qtablewidgetitem1010->setText(QCoreApplication::translate("InputRZForm", "28", nullptr));
        QTableWidgetItem *___qtablewidgetitem1011 = SpecPlotTable->verticalHeaderItem(28);
        ___qtablewidgetitem1011->setText(QCoreApplication::translate("InputRZForm", "29", nullptr));
        QTableWidgetItem *___qtablewidgetitem1012 = SpecPlotTable->verticalHeaderItem(29);
        ___qtablewidgetitem1012->setText(QCoreApplication::translate("InputRZForm", "30", nullptr));
        QTableWidgetItem *___qtablewidgetitem1013 = SpecPlotTable->verticalHeaderItem(30);
        ___qtablewidgetitem1013->setText(QCoreApplication::translate("InputRZForm", "31", nullptr));
        QTableWidgetItem *___qtablewidgetitem1014 = SpecPlotTable->verticalHeaderItem(31);
        ___qtablewidgetitem1014->setText(QCoreApplication::translate("InputRZForm", "32", nullptr));
        QTableWidgetItem *___qtablewidgetitem1015 = SpecPlotTable->verticalHeaderItem(32);
        ___qtablewidgetitem1015->setText(QCoreApplication::translate("InputRZForm", "33", nullptr));
        QTableWidgetItem *___qtablewidgetitem1016 = SpecPlotTable->verticalHeaderItem(33);
        ___qtablewidgetitem1016->setText(QCoreApplication::translate("InputRZForm", "34", nullptr));
        QTableWidgetItem *___qtablewidgetitem1017 = SpecPlotTable->verticalHeaderItem(34);
        ___qtablewidgetitem1017->setText(QCoreApplication::translate("InputRZForm", "35", nullptr));
        QTableWidgetItem *___qtablewidgetitem1018 = SpecPlotTable->verticalHeaderItem(35);
        ___qtablewidgetitem1018->setText(QCoreApplication::translate("InputRZForm", "36", nullptr));
        QTableWidgetItem *___qtablewidgetitem1019 = SpecPlotTable->verticalHeaderItem(36);
        ___qtablewidgetitem1019->setText(QCoreApplication::translate("InputRZForm", "37", nullptr));
        QTableWidgetItem *___qtablewidgetitem1020 = SpecPlotTable->verticalHeaderItem(37);
        ___qtablewidgetitem1020->setText(QCoreApplication::translate("InputRZForm", "38", nullptr));
        QTableWidgetItem *___qtablewidgetitem1021 = SpecPlotTable->verticalHeaderItem(38);
        ___qtablewidgetitem1021->setText(QCoreApplication::translate("InputRZForm", "39", nullptr));
        QTableWidgetItem *___qtablewidgetitem1022 = SpecPlotTable->verticalHeaderItem(39);
        ___qtablewidgetitem1022->setText(QCoreApplication::translate("InputRZForm", "40", nullptr));
        QTableWidgetItem *___qtablewidgetitem1023 = SpecPlotTable->verticalHeaderItem(40);
        ___qtablewidgetitem1023->setText(QCoreApplication::translate("InputRZForm", "41", nullptr));
        QTableWidgetItem *___qtablewidgetitem1024 = SpecPlotTable->verticalHeaderItem(41);
        ___qtablewidgetitem1024->setText(QCoreApplication::translate("InputRZForm", "42", nullptr));
        QTableWidgetItem *___qtablewidgetitem1025 = SpecPlotTable->verticalHeaderItem(42);
        ___qtablewidgetitem1025->setText(QCoreApplication::translate("InputRZForm", "43", nullptr));
        QTableWidgetItem *___qtablewidgetitem1026 = SpecPlotTable->verticalHeaderItem(43);
        ___qtablewidgetitem1026->setText(QCoreApplication::translate("InputRZForm", "44", nullptr));
        QTableWidgetItem *___qtablewidgetitem1027 = SpecPlotTable->verticalHeaderItem(44);
        ___qtablewidgetitem1027->setText(QCoreApplication::translate("InputRZForm", "45", nullptr));
        QTableWidgetItem *___qtablewidgetitem1028 = SpecPlotTable->verticalHeaderItem(45);
        ___qtablewidgetitem1028->setText(QCoreApplication::translate("InputRZForm", "46", nullptr));
        QTableWidgetItem *___qtablewidgetitem1029 = SpecPlotTable->verticalHeaderItem(46);
        ___qtablewidgetitem1029->setText(QCoreApplication::translate("InputRZForm", "47", nullptr));
        QTableWidgetItem *___qtablewidgetitem1030 = SpecPlotTable->verticalHeaderItem(47);
        ___qtablewidgetitem1030->setText(QCoreApplication::translate("InputRZForm", "48", nullptr));
        QTableWidgetItem *___qtablewidgetitem1031 = SpecPlotTable->verticalHeaderItem(48);
        ___qtablewidgetitem1031->setText(QCoreApplication::translate("InputRZForm", "49", nullptr));
        QTableWidgetItem *___qtablewidgetitem1032 = SpecPlotTable->verticalHeaderItem(49);
        ___qtablewidgetitem1032->setText(QCoreApplication::translate("InputRZForm", "50", nullptr));
        TabWidgetRZ->setTabText(TabWidgetRZ->indexOf(PLOTtab), QCoreApplication::translate("InputRZForm", "Plot Control", nullptr));
    } // retranslateUi

};

namespace Ui {
    class InputRZForm: public Ui_InputRZForm {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_INPUTRZ_H
