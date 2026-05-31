// ---- Polyglot Opening Book Reader for Datagen ----
// Reads standard Polyglot .bin files and returns weighted-random moves.

use crate::types::*;
use crate::board::Position;
use crate::movegen;
use crate::attacks;


// ---- Polyglot Zobrist randoms (781 entries) ----
// Layout: piece[12] * square[64] = 768 + castling[4] + enpassant[8] + turn[1]
// Polyglot piece order: BP, WP, BN, WN, BB, WB, BR, WR, BQ, WQ, BK, WK
#[rustfmt::skip]
static POLY_RANDOMS: [u64; 781] = [
    0x9D39247E33776D41, 0x2AF7398005AAA5C7, 0x44DB015024623547, 0x9C15F73E62A76AE2,
    0x75834465489C0C89, 0x3290AC3A203001BF, 0x0FBBAD1F61042279, 0xE83A908FF2FB60CA,
    0x0D7E765D58755C10, 0x1A083822CEAFE02D, 0x9605D5F0E25EC3B0, 0xD021FF5CD13A2ED5,
    0x40BDF15D4A672E32, 0x011355146FD56395, 0x5DB4832046F3D9E5, 0x239F8B2D7FF719CC,
    0x05D1A1AE85B49AA1, 0x679F848F6E8FC971, 0x7449BBFF801FED0B, 0x7D11CDB1C3B7ADF0,
    0x82C7709E781EB7CC, 0xF3218F1C9510786C, 0x331478F3AF51BBE6, 0x4BB38DE5E7219443,
    0xAA649C6EBCFD50FC, 0x8DBD98A352AFD40B, 0x87D2074B81D79217, 0x19F3C751D3E92AE1,
    0xB4AB30F062B19ABF, 0x7B0500AC42047AC4, 0xC9452CA81A09D85D, 0x24AA6C514DA27500,
    0x4C9F34427501B447, 0x14A68FD73C910841, 0xA71B9B83461CBD93, 0x03488B95B0F1850F,
    0x637B2B34FF93C040, 0x09D1BC9A3DD90A94, 0x3575668334A1DD3B, 0x735E2B97A4C45A23,
    0x18727070F1BD400B, 0x1FCBACD259BF02E7, 0xD310A7C2CE9B6555, 0xBF983FE0FE5D8244,
    0x9F74D14F7454A824, 0x51EBDC4AB9BA3035, 0x5C82C505DB9AB0FA, 0xFCF7FE8A3430B241,
    0x3253A729B9BA3DDE, 0x8C74C368081B3075, 0xB9BC6C87167C33E7, 0x7EF48F2B83024E20,
    0x11D505D4C351BD7F, 0x6568FCA92C76A243, 0x4DE0B0F40F32A7B8, 0x96D693460CC37E5D,
    0x42E240CB63689F2F, 0x6D2BDCDAE2919661, 0x42880B0236E4D951, 0x5F0F4A5898171BB6,
    0x39F890F579F92F88, 0x93C5B5F47356388B, 0x63DC359D8D231B78, 0xEC16CA8AEA98AD76,
    0x5355F900C2A82DC7, 0x07FB9F855A997142, 0x5093417AA8A7ED5E, 0x7BCBC38DA25A7F3C,
    0x19FC8A768CF4B6D4, 0x637A7780DECFC0D9, 0x8249A47AEE0E41F7, 0x79AD695501E7D1E8,
    0x14ACBAF4777D5776, 0xF145B6BECCDEA195, 0xDABF2AC8201752FC, 0x24C3C94DF9C8D3F6,
    0xBB6E2924F03912EA, 0x0CE26C0B95C980D9, 0xA49CD132BFBF7CC4, 0xE99D662AF4243939,
    0x27E6AD7891165C3F, 0x8535F040B9744FF1, 0x54B3F4FA5F40D873, 0x72B12C32127FED2B,
    0xEE954D3C7B411F47, 0x9A85AC909A24EAA1, 0x70AC4CD9F04F21F5, 0xF9B89D3E99A075C2,
    0x87B3E2B2B5C907B1, 0xA366E5B8C54F48B8, 0xAE4A9346CC3F7CF2, 0x1920C04D47267BBD,
    0x87BF02C6B49E2AE9, 0x092237AC237F3859, 0xFF07F64EF8ED14D0, 0x8DE8DCA9F03CC54E,
    0x9C1633264DB49C89, 0xB3F22C3D0B0B38ED, 0x390E5FB44D01144B, 0x5BFEA5B4712768E9,
    0x1E1032911FA78984, 0x9A74ACB964E78CB3, 0x4F80F7A035DAFB04, 0x6304D09A0B3738C4,
    0x2171E64683023A08, 0x5B9B63EB9CEFF80C, 0x506AACF489889342, 0x1881AFC9A3A701D6,
    0x6503080440750644, 0xDFD395339CDBF4A7, 0xEF927DBCF00C20F2, 0x7B32F7D1E03680EC,
    0xB9FD7620E7316243, 0x05A7E8A57DB91B77, 0xB5889C6E15630A75, 0x4A750A09CE9573F7,
    0xCF464CEC899A2F8A, 0xF538639CE705B824, 0x3C79A0FF5580EF7F, 0xEDE6C87F8477609D,
    0x799E81F05BC93F31, 0x86536B8CF3428A8C, 0x97D7374C60087B73, 0xA246637CFF328532,
    0x043FCAE60CC0EBA0, 0x920E449535DD359E, 0x70EB093B15B290CC, 0x73A1921916591CBD,
    0x56436C9FE1A1AA8D, 0xEFAC4B70633B8F81, 0xBB215798D45DF7AF, 0x45F20042F24F1768,
    0x930F80F4E8EB7462, 0xFF6712FFCFD75EA1, 0xAE623FD67468AA70, 0xDD2C5BC84BC8D8FC,
    0x7EED120D54CF2DD9, 0x22FE545401165F1C, 0xC91800E98FB99929, 0x808BD68E6AC10365,
    0xDEC468145B7605F6, 0x1BEDE3A3AEF53302, 0x43539603D6C55602, 0xAA969B5C691CCB7A,
    0xA87832D392EFEE56, 0x65942C7B3C7E11AE, 0xDED2D633CAD004F6, 0x21F08570F420E565,
    0xB415938D7DA94E3C, 0x91B859E59ECB6350, 0x10CFF333E0ED804A, 0x28AED140BE0BB7DD,
    0xC5CC1D89724FA456, 0x5648F680F11A2741, 0x2D255069F0B7DAB3, 0x9BC5A38EF729ABD4,
    0xEF2F054308F6A2BC, 0xAF2042F5CC5C2858, 0x480412BAB7F5BE2A, 0xAEF3AF4A563DFE43,
    0x19AFE59AE451497F, 0x52593803DFF1E840, 0xF4F076E65F2CE6F0, 0x11379625747D5AF3,
    0xBCE5D2248682C115, 0x9DA4243DE836994F, 0x066F70B33FE09017, 0x4DC4DE189B671A1C,
    0x51039AB7712457C3, 0xC07A3F80C31FB4B4, 0xB46EE9C5E64A6E7C, 0xB3819A42ABE61C87,
    0x21A007933A522A20, 0x2DF16F761598AA4F, 0x763C4A1371B368FD, 0xF793C46702E086A0,
    0xD7288E012AEB8D31, 0xDE336A2A4BC1C44B, 0x0BF692B38D079F23, 0x2C604A7A177326B3,
    0x4850E73E03EB6064, 0xCFC447F1E53C8E1B, 0xB05CA3F564268D99, 0x9AE182C8BC9474E8,
    0xA4FC4BD4FC5558CA, 0xE755178D58FC4E76, 0x69B97DB1A4C03DFE, 0xF9B5B7C4ACC67C96,
    0xFC6A82D64B8655FB, 0x9C684CB6C4D24417, 0x8EC97D2917456ED0, 0x6703DF9D2924E97E,
    0xC547F57E42A7444E, 0x78E37644E7CAD29E, 0xFE9A44E9362F05FA, 0x08BD35CC38336615,
    0x9315E5EB3A129ACE, 0x94061B871E04DF75, 0xDF1D9F9D784BA010, 0x3BBA57B68871B59D,
    0xD2B7ADEEDED1F73F, 0xF7A255D83BC373F8, 0xD7F4F2448C0CEB81, 0xD95BE88CD210FFA7,
    0x336F52F8FF4728E7, 0xA74049DAC312AC71, 0xA2F61BB6E437FDB5, 0x4F2A5CB07F6A35B3,
    0x87D380BDA5BF7859, 0x16B9F7E06C453A21, 0x7BA2484C8A0FD54E, 0xF3A678CAD9A2E38C,
    0x39B0BF7DDE437BA2, 0xFCAF55C1BF8A4424, 0x18FCF680573FA594, 0x4C0563B89F495AC3,
    0x40E087931A00930D, 0x8CFFA9412EB642C1, 0x68CA39053261169F, 0x7A1EE967D27579E2,
    0x9D1D60E5076F5B6F, 0x3810E399B6F65BA2, 0x32095B6D4AB5F9B1, 0x35CAB62109DD038A,
    0xA90B24499FCFAFB1, 0x77A225A07CC2C6BD, 0x513E5E634C70E331, 0x4361C0CA3F692F12,
    0xD941ACA44B20A45B, 0x528F7C8602C5807B, 0x52AB92BEB9613989, 0x9D1DFA2EFC557F73,
    0x722FF175F572C348, 0x1D1260A51107FE97, 0x7A249A57EC0C9BA2, 0x04208FE9E8F7F2D6,
    0x5A110C6058B920A0, 0x0CD9A497658A5698, 0x56FD23C8F9715A4C, 0x284C847B9D887AAE,
    0x04FEABFBBDB619CB, 0x742E1E651C60BA83, 0x9A9632E65904AD3C, 0x881B82A13B51B9E2,
    0x506E6744CD974924, 0xB0183DB56FFC6A79, 0x0ED9B915C66ED37E, 0x5E11E86D5873D484,
    0xF678647E3519AC6E, 0x1B85D488D0F20CC5, 0xDAB9FE6525D89021, 0x0D151D86ADB73615,
    0xA865A54EDCC0F019, 0x93C42566AEF98FFB, 0x99E7AFEABE000731, 0x48CBFF086DDF285A,
    0x7F9B6AF1EBF78BAF, 0x58627E1A149BBA21, 0x2CD16E2ABD791E33, 0xD363EFF5F0977996,
    0x0CE2A38C344A6EED, 0x1A804AADB9CFA741, 0x907F30421D78C5DE, 0x501F65EDB3034D07,
    0x37624AE5A48FA6E9, 0x957BAF61700CFF4E, 0x3A6C27934E31188A, 0xD49503536ABCA345,
    0x088E049589C432E0, 0xF943AEE7FEBF21B8, 0x6C3B8E3E336139D3, 0x364F6FFA464EE52E,
    0xD60F6DCEDC314222, 0x56963B0DCA418FC0, 0x16F50EDF91E513AF, 0xEF1955914B609F93,
    0x565601C0364E3228, 0xECB53939887E8175, 0xBAC7A9A18531294B, 0xB344C470397BBA52,
    0x65D34954DAF3CEBD, 0xB4B81B3FA97511E2, 0xB422061193D6F6A7, 0x071582401C38434D,
    0x7A13F18BBEDC4FF5, 0xBC4097B116C524D2, 0x59B97885E2F2EA28, 0x99170A5DC3115544,
    0x6F423357E7C6A9F9, 0x325928EE6E6F8794, 0xD0E4366228B03343, 0x565C31F7DE89EA27,
    0x30F5611484119414, 0xD873DB391292ED4F, 0x7BD94E1D8E17DEBC, 0xC7D9F16864A76E94,
    0x947AE053EE56E63C, 0xC8C93882F9475F5F, 0x3A9BF55BA91F81CA, 0xD9A11FBB3D9808E4,
    0x0FD22063EDC29FCA, 0xB3F256D8ACA0B0B9, 0xB03031A8B4516E84, 0x35DD37D5871448AF,
    0xE9F6082B05542E4E, 0xEBFAFA33D7254B59, 0x9255ABB50D532280, 0xB9AB4CE57F2D34F3,
    0x693501D628297551, 0xC62C58F97DD949BF, 0xCD454F8F19C5126A, 0xBBE83F4ECC2BDECB,
    0xDC842B7E2819E230, 0xBA89142E007503B8, 0xA3BC941D0A5061CB, 0xE9F6760E32CD8021,
    0x09C7E552BC76492F, 0x852F54934DA55CC9, 0x8107FCCF064FCF56, 0x098954D51FFF6580,
    0x23B70EDB1955C4BF, 0xC330DE426430F69D, 0x4715ED43E8A45C0A, 0xA8D7E4DAB780A08D,
    0x0572B974F03CE0BB, 0xB57D2E985E1419C7, 0xE8D9ECBE2CF3D73F, 0x2FE4B17170E59750,
    0x11317BA87905E790, 0x7FBF21EC8A1F45EC, 0x1725CABFCB045B00, 0x964E915CD5E2B207,
    0x3E2B8BCBF016D66D, 0xBE7444E39328A0AC, 0xF85B2B4FBCDE44B7, 0x49353FEA39BA63B1,
    0x1DD01AAFCD53486A, 0x1FCA8A92FD719F85, 0xFC7C95D827357AFA, 0x18A6A990C8B35EBD,
    0xCCCB7005C6B9C28D, 0x3BDBB92C43B17F26, 0xAA70B5B4F89695A2, 0xE94C39A54A98307F,
    0xB7A0B174CFF6F36E, 0xD4DBA84729AF48AD, 0x2E18BC1AD9704A68, 0x2DE0966DAF2F8B1C,
    0xB9C11D5B1E43A07E, 0x64972D68DEE33360, 0x94628D38D0C20584, 0xDBC0D2B6AB90A559,
    0xD2733C4335C6A72F, 0x7E75D99D94A70F4D, 0x6CED1983376FA72B, 0x97FCAACBF030BC24,
    0x7B77497B32503B12, 0x8547EDDFB81CCB94, 0x79999CDFF70902CB, 0xCFFE1939438E9B24,
    0x829626E3892D95D7, 0x92FAE24291F2B3F1, 0x63E22C147B9C3403, 0xC678B6D860284A1C,
    0x5873888850659AE7, 0x0981DCD296A8736D, 0x9F65789A6509A440, 0x9FF38FED72E9052F,
    0xE479EE5B9930578C, 0xE7F28ECD2D49EECD, 0x56C074A581EA17FE, 0x5544F7D774B14AEF,
    0x7B3F0195FC6F290F, 0x12153635B2C0CF57, 0x7F5126DBBA5E0CA7, 0x7A76956C3EAFB413,
    0x3D5774A11D31AB39, 0x8A1B083821F40CB4, 0x7B4A38E32537DF62, 0x950113646D1D6E03,
    0x4DA8979A0041E8A9, 0x3BC36E078F7515D7, 0x5D0A12F27AD310D1, 0x7F9D1A2E1EBE1327,
    0xDA3A361B1C5157B1, 0xDCDD7D20903D0C25, 0x36833336D068F707, 0xCE68341F79893389,
    0xAB9090168DD05F34, 0x43954B3252DC25E5, 0xB438C2B67F98E5E9, 0x10DCD78E3851A492,
    0xDBC27AB5447822BF, 0x9B3CDB65F82CA382, 0xB67B7896167B4C84, 0xBFCED1B0048EAC50,
    0xA9119B60369FFEBD, 0x1FFF7AC80904BF45, 0xAC12FB171817EEE7, 0xAF08DA9177DDA93D,
    0x1B0CAB936E65C744, 0xB559EB1D04E5E932, 0xC37B45B3F8D6F2BA, 0xC3A9DC228CAAC9E9,
    0xF3B8B6675A6507FF, 0x9FC477DE4ED681DA, 0x67378D8ECCEF96CB, 0x6DD856D94D259236,
    0xA319CE15B0B4DB31, 0x073973751F12DD5E, 0x8A8E849EB32781A5, 0xE1925C71285279F5,
    0x74C04BF1790C0EFE, 0x4DDA48153C94938A, 0x9D266D6A1CC0542C, 0x7440FB816508C4FE,
    0x13328503DF48229F, 0xD6BF7BAEE43CAC40, 0x4838D65F6EF6748F, 0x1E152328F3318DEA,
    0x8F8419A348F296BF, 0x72C8834A5957B511, 0xD7A023A73260B45C, 0x94EBC8ABCFB56DAE,
    0x9FC10D0F989993E0, 0xDE68A2355B93CAE6, 0xA44CFE79AE538BBE, 0x9D1D84FCCE371425,
    0x51D2B1AB2DDFB636, 0x2FD7E4B9E72CD38C, 0x65CA5B96B7552210, 0xDD69A0D8AB3B546D,
    0x604D51B25FBF70E2, 0x73AA8A564FB7AC9E, 0x1A8C1E992B941148, 0xAAC40A2703D9BEA0,
    0x764DBEAE7FA4F3A6, 0x1E99B96E70A9BE8B, 0x2C5E9DEB57EF4743, 0x3A938FEE32D29981,
    0x26E6DB8FFDF5ADFE, 0x469356C504EC9F9D, 0xC8763C5B08D1908C, 0x3F6C6AF859D80055,
    0x7F7CC39420A3A545, 0x9BFB227EBDF4C5CE, 0x89039D79D6FC5C5C, 0x8FE88B57305E2AB6,
    0xA09E8C8C35AB96DE, 0xFA7E393983325753, 0xD6B6D0ECC617C699, 0xDFEA21EA9E7557E3,
    0xB67C1FA481680AF8, 0xCA1E3785A9E724E5, 0x1CFC8BED0D681639, 0xD18D8549D140CAEA,
    0x4ED0FE7E9DC91335, 0xE4DBF0634473F5D2, 0x1761F93A44D5AEFE, 0x53898E4C3910DA55,
    0x734DE8181F6EC39A, 0x2680B122BAA28D97, 0x298AF231C85BAFAB, 0x7983EED3740847D5,
    0x66C1A2A1A60CD889, 0x9E17E49642A3E4C1, 0xEDB454E7BADC0805, 0x50B704CAB602C329,
    0x4CC317FB9CDDD023, 0x66B4835D9EAFEA22, 0x219B97E26FFC81BD, 0x261E4E4C0A333A9D,
    0x1FE2CCA76517DB90, 0xD7504DFA8816EDBB, 0xB9571FA04DC089C8, 0x1DDC0325259B27DE,
    0xCF3F4688801EB9AA, 0xF4F5D05C10CAB243, 0x38B6525C21A42B0E, 0x36F60E2BA4FA6800,
    0xEB3593803173E0CE, 0x9C4CD6257C5A3603, 0xAF0C317D32ADAA8A, 0x258E5A80C7204C4B,
    0x8B889D624D44885D, 0xF4D14597E660F855, 0xD4347F66EC8941C3, 0xE699ED85B0DFB40D,
    0x2472F6207C2D0484, 0xC2A1E7B5B459AEB5, 0xAB4F6451CC1D45EC, 0x63767572AE3D6174,
    0xA59E0BD101731A28, 0x116D0016CB948F09, 0x2CF9C8CA052F6E9F, 0x0B090A7560A968E3,
    0xABEEDDB2DDE06FF1, 0x58EFC10B06A2068D, 0xC6E57A78FBD986E0, 0x2EAB8CA63CE802D7,
    0x14A195640116F336, 0x7C0828DD624EC390, 0xD74BBE77E6116AC7, 0x804456AF10F5FB53,
    0xEBE9EA2ADF4321C7, 0x03219A39EE587A30, 0x49787FEF17AF9924, 0xA1E9300CD8520548,
    0x5B45E522E4B1B4EF, 0xB49C3B3995091A36, 0xD4490AD526F14431, 0x12A8F216AF9418C2,
    0x001F837CC7350524, 0x1877B51E57A764D5, 0xA2853B80F17F58EE, 0x993E1DE72D36D310,
    0xB3598080CE64A656, 0x252F59CF0D9F04BB, 0xD23C8E176D113600, 0x1BDA0492E7E4586E,
    0x21E0BD5026C619BF, 0x3B097ADAF088F94E, 0x8D14DEDB30BE846E, 0xF95CFFA23AF5F6F4,
    0x3871700761B3F743, 0xCA672B91E9E4FA16, 0x64C8E531BFF53B55, 0x241260ED4AD1E87D,
    0x106C09B972D2E822, 0x7FBA195410E5CA30, 0x7884D9BC6CB569D8, 0x0647DFEDCD894A29,
    0x63573FF03E224774, 0x4FC8E9560F91B123, 0x1DB956E450275779, 0xB8D91274B9E9D4FB,
    0xA2EBEE47E2FBFCE1, 0xD9F1F30CCD97FB09, 0xEFED53D75FD64E6B, 0x2E6D02C36017F67F,
    0xA9AA4D20DB084E9B, 0xB64BE8D8B25396C1, 0x70CB6AF7C2D5BCF0, 0x98F076A4F7A2322E,
    0xBF84470805E69B5F, 0x94C3251F06F90CF3, 0x3E003E616A6591E9, 0xB925A6CD0421AFF3,
    0x61BDD1307C66E300, 0xBF8D5108E27E0D48, 0x240AB57A8B888B20, 0xFC87614BAF287E07,
    0xEF02CDD06FFDB432, 0xA1082C0466DF6C0A, 0x8215E577001332C8, 0xD39BB9C3A48DB6CF,
    0x2738259634305C14, 0x61CF4F94C97DF93D, 0x1B6BACA2AE4E125B, 0x758F450C88572E0B,
    0x959F587D507A8359, 0xB063E962E045F54D, 0x60E8ED72C0DFF5D1, 0x7B64978555326F9F,
    0xFD080D236DA814BA, 0x8C90FD9B083F4558, 0x106F72FE81E2C590, 0x7976033A39F7D952,
    0xA4EC0132764CA04B, 0x733EA705FAE4FA77, 0xB4D8F77BC3E56167, 0x9E21F4F903B33FD9,
    0x9D765E419FB69F6D, 0xD30C088BA61EA5EF, 0x5D94337FBFAF7F5B, 0x1A4E4822EB4D7A59,
    0x6FFE73E81B637FB3, 0xDDF957BC36D8B9CA, 0x64D0E29EEA8838B3, 0x08DD9BDFD96B9F63,
    0x087E79E5A57D1D13, 0xE328E230E3E2B3FB, 0x1C2559E30F0946BE, 0x720BF5F26F4D2EAA,
    0xB0774D261CC609DB, 0x443F64EC5A371195, 0x4112CF68649A260E, 0xD813F2FAB7F5C5CA,
    0x660D3257380841EE, 0x59AC2C7873F910A3, 0xE846963877671A17, 0x93B633ABFA3469F8,
    0xC0C0F5A60EF4CDCF, 0xCAF21ECD4377B28C, 0x57277707199B8175, 0x506C11B9D90E8B1D,
    0xD83CC2687A19255F, 0x4A29C6465A314CD1, 0xED2DF21216235097, 0xB5635C95FF7296E2,
    0x22AF003AB672E811, 0x52E762596BF68235, 0x9AEBA33AC6ECC6B0, 0x944F6DE09134DFB6,
    0x6C47BEC883A7DE39, 0x6AD047C430A12104, 0xA5B1CFDBA0AB4067, 0x7C45D833AFF07862,
    0x5092EF950A16DA0B, 0x9338E69C052B8E7B, 0x455A4B4CFE30E3F5, 0x6B02E63195AD0CF8,
    0x6B17B224BAD6BF27, 0xD1E0CCD25BB9C169, 0xDE0C89A556B9AE70, 0x50065E535A213CF6,
    0x9C1169FA2777B874, 0x78EDEFD694AF1EED, 0x6DC93D9526A50E68, 0xEE97F453F06791ED,
    0x32AB0EDB696703D3, 0x3A6853C7E70757A7, 0x31865CED6120F37D, 0x67FEF95D92607890,
    0x1F2B1D1F15F6DC9C, 0xB69E38A8965C6B65, 0xAA9119FF184CCCF4, 0xF43C732873F24C13,
    0xFB4A3D794A9A80D2, 0x3550C2321FD6109C, 0x371F77E76BB8417E, 0x6BFA9AAE5EC05779,
    0xCD04F3FF001A4778, 0xE3273522064480CA, 0x9F91508BFFCFC14A, 0x049A7F41061A9E60,
    0xFCB6BE43A9F2FE9B, 0x08DE8A1C7797DA9B, 0x8F9887E6078735A1, 0xB5B4071DBFC73A66,
    0x230E343DFBA08D33, 0x43ED7F5A0FAE657D, 0x3A88A0FBBCB05C63, 0x21874B8B4D2DBC4F,
    0x1BDEA12E35F6A8C9, 0x53C065C6C8E63528, 0xE34A1D250E7A8D6B, 0xD6B04D3B7651DD7E,
    0x5E90277E7CB39E2D, 0x2C046F22062DC67D, 0xB10BB459132D0A26, 0x3FA9DDFB67E2F199,
    0x0E09B88E1914F7AF, 0x10E8B35AF3EEAB37, 0x9EEDECA8E272B933, 0xD4C718BC4AE8AE5F,
    0x81536D601170FC20, 0x91B534F885818A06, 0xEC8177F83F900978, 0x190E714FADA5156E,
    0xB592BF39B0364963, 0x89C350C893AE7DC1, 0xAC042E70F8B383F2, 0xB49B52E587A1EE60,
    0xFB152FE3FF26DA89, 0x3E666E6F69AE2C15, 0x3B544EBE544C19F9, 0xE805A1E290CF2456,
    0x24B33C9D7ED25117, 0xE74733427B72F0C1, 0x0A804D18B7097475, 0x57E3306D881EDB4F,
    0x4AE7D6A36EB5DBCB, 0x2D8D5432157064C8, 0xD1E649DE1E7F268B, 0x8A328A1CEDFE552C,
    0x07A3AEC79624C7DA, 0x84547DDC3E203C94, 0x990A98FD5071D263, 0x1A4FF12616EEFC89,
    0xF6F7FD1431714200, 0x30C05B1BA332F41C, 0x8D2636B81555A786, 0x46C9FEB55D120902,
    0xCCEC0A73B49C9921, 0x4E9D2827355FC492, 0x19EBB029435DCB0F, 0x4659D2B743848A2C,
    0x963EF2C96B33BE31, 0x74F85198B05A2E7D, 0x5A0F544DD2B1FB18, 0x03727073C2E134B1,
    0xC7F6AA2DE59AEA61, 0x352787BAA0D7C22F, 0x9853EAB63B5E0B35, 0xABBDCDD7ED5C0860,
    0xCF05DAF5AC8D77B0, 0x49CAD48CEBF4A71E, 0x7A4C10EC2158C4A6, 0xD9E92AA246BF719E,
    0x13AE978D09FE5557, 0x730499AF921549FF, 0x4E4B705B92903BA4, 0xFF577222C14F0A3A,
    0x55B6344CF97AAFAE, 0xB862225B055B6960, 0xCAC09AFBDDD2CDB4, 0xDAF8E9829FE96B5F,
    0xB5FDFC5D3132C498, 0x310CB380DB6F7503, 0xE87FBB46217A360E, 0x2102AE466EBB1148,
    0xF8549E1A3AA5E00D, 0x07A69AFDCC42261A, 0xC4C118BFE78FEAAE, 0xF9F4892ED96BD438,
    0x1AF3DBE25D8F45DA, 0xF5B4B0B0D2DEEEB4, 0x962ACEEFA82E1C84, 0x046E3ECAAF453CE9,
    0xF05D129681949A4C, 0x964781CE734B3C84, 0x9C2ED44081CE5FBD, 0x522E23F3925E319E,
    0x177E00F9FC32F791, 0x2BC60A63A6F3B3F2, 0x222BBFAE61725606, 0x486289DDCC3D6780,
    0x7DC7785B8EFDFC80, 0x8AF38731C02BA980, 0x1FAB64EA29A2DDF7, 0xE4D9429322CD065A,
    0x9DA058C67844F20C, 0x24C0E332B70019B0, 0x233003B5A6CFE6AD, 0xD586BD01C5C217F6,
    0x5E5637885F29BC2B, 0x7EBA726D8C94094B, 0x0A56A5F0BFE39272, 0xD79476A84EE20D06,
    0x9E4C1269BAA4BF37, 0x17EFEE45B0DEE640, 0x1D95B0A5FCF90BC6, 0x93CBE0B699C2585D,
    0x65FA4F227A2B6D79, 0xD5F9E858292504D5, 0xC2B5A03F71471A6F, 0x59300222B4561E00,
    0xCE2F8642CA0712DC, 0x7CA9723FBB2E8988, 0x2785338347F2BA08, 0xC61BB3A141E50E8C,
    0x150F361DAB9DEC26, 0x9F6A419D382595F4, 0x64A53DC924FE7AC9, 0x142DE49FFF7A7C3D,
    0x0C335248857FA9E7, 0x0A9C32D5EAE45305, 0xE6C42178C4BBB92E, 0x71F1CE2490D20B07,
    0xF1BCC3D275AFE51A, 0xE728E8C83C334074, 0x96FBF83A12884624, 0x81A1549FD6573DA5,
    0x5FA7867CAF35E149, 0x56986E2EF3ED091B, 0x917F1DD5F8886C61, 0xD20D8C88C8FFE65F,
    0x31D71DCE64B2C310, 0xF165B587DF898190, 0xA57E6339DD2CF3A0, 0x1EF6E6DBB1961EC9,
    0x70CC73D90BC26E24, 0xE21A6B35DF0C3AD7, 0x003A93D8B2806962, 0x1C99DED33CB890A1,
    0xCF3145DE0ADD4289, 0xD0E4427A5514FB72, 0x77C621CC9FB3A483, 0x67A34DAC4356550B,
    0xF8D626AAAF278509,
];

/// Compute Polyglot hash for a clrsrc Position.
pub fn polyglot_hash(pos: &Position) -> u64 {
    let mut hash: u64 = 0;

    // Piece placement: Polyglot order is BP, WP, BN, WN, BB, WB, BR, WR, BQ, WQ, BK, WK
    // clrsrc: mailbox[sq] = color*6 + piecetype, where Pawn=0..King=5
    // Polyglot piece index: piecetype * 2 + (if white then 1 else 0)
    for sq in 0..64u8 {
        let piece = pos.mailbox[sq as usize];
        if piece == EMPTY {
            continue;
        }
        let color = piece_color(piece);
        let pt = piece_type(piece);
        let poly_piece = pt.index() * 2 + if color == WHITE { 1 } else { 0 };
        hash ^= POLY_RANDOMS[64 * poly_piece + sq as usize];
    }

    // Castling (4 entries at offset 768)
    if pos.castling.has(CastlingRights::WK) {
        hash ^= POLY_RANDOMS[768];
    }
    if pos.castling.has(CastlingRights::WQ) {
        hash ^= POLY_RANDOMS[769];
    }
    if pos.castling.has(CastlingRights::BK) {
        hash ^= POLY_RANDOMS[770];
    }
    if pos.castling.has(CastlingRights::BQ) {
        hash ^= POLY_RANDOMS[771];
    }

    // En passant (8 entries at offset 772, indexed by file). Polyglot spec: the ep key is only
    // mixed when a pawn of the side to move can actually capture on the ep square. clrsrc's
    // Position::make_move sets ep_square on every double-pawn-push without this check, so without
    // the gate below clrsrc would hash positions differently from any Polyglot-conformant builder
    // (e.g. the jugernaut v4 book): the engine would silently miss its own book after every 2-step
    // pawn push on the main line.
    if pos.ep_square != NO_SQ {
        let ep_file = file_of(pos.ep_square) as i32;
        let captured_rank: i32 = if pos.side == WHITE { 4 } else { 3 };
        let capturer = make_piece(pos.side, PAWN);
        let mut capturable = false;
        if ep_file > 0 {
            let sq = (captured_rank * 8 + ep_file - 1) as usize;
            if pos.mailbox[sq] == capturer { capturable = true; }
        }
        if !capturable && ep_file < 7 {
            let sq = (captured_rank * 8 + ep_file + 1) as usize;
            if pos.mailbox[sq] == capturer { capturable = true; }
        }
        if capturable {
            hash ^= POLY_RANDOMS[772 + ep_file as usize];
        }
    }

    // Turn: Polyglot XORs the turn key when WHITE to move
    if pos.side == WHITE {
        hash ^= POLY_RANDOMS[780];
    }

    hash
}

/// Opening book — reads Polyglot .bin directly from disk via binary search.
/// No preloading needed: the file is sorted by hash, so we seek + read only matching entries.
pub struct Book {
    data: Vec<u8>, // entire file in memory (fast, no per-probe I/O)
    num_entries: usize,
}

impl Book {
    /// Open a Polyglot .bin book. Reads entire file into memory but is instant for typical books.
    /// For very large books (>100MB), still fast since it's a single sequential read.
    pub fn load(path: &str) -> Option<Book> {
        let data = std::fs::read(path).ok()?;
        // Polyglot entries are exactly 16 bytes; a non-multiple length means a truncated
        // file and the trailing partial entry is silently dropped by this division.
        debug_assert!(data.len() % 16 == 0, "Polyglot book {} has a truncated trailing entry", path);
        let num_entries = data.len() / 16;
        eprintln!("book: opened {} ({} entries)", path, num_entries);
        Some(Book { data, num_entries })
    }

    /// Read entry at index
    #[inline]
    fn entry_key(&self, idx: usize) -> u64 {
        let off = idx * 16;
        u64::from_be_bytes(self.data[off..off + 8].try_into().unwrap())
    }

    #[inline]
    fn entry_move(&self, idx: usize) -> u16 {
        let off = idx * 16 + 8;
        u16::from_be_bytes(self.data[off..off + 2].try_into().unwrap())
    }

    #[inline]
    fn entry_weight(&self, idx: usize) -> u16 {
        let off = idx * 16 + 10;
        u16::from_be_bytes(self.data[off..off + 2].try_into().unwrap())
    }

    /// Binary search for first entry with given hash key.
    fn find_first(&self, hash: u64) -> Option<usize> {
        let mut lo = 0usize;
        let mut hi = self.num_entries;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.entry_key(mid) < hash {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo < self.num_entries && self.entry_key(lo) == hash {
            Some(lo)
        } else {
            None
        }
    }

    /// Pick a book move for the position.
    /// If `best_only` is true, always pick the highest-weight move.
    /// Otherwise, weighted random selection.
    pub fn probe(&self, pos: &mut Position, rng_val: u64, best_only: bool) -> Option<Move> {
        let hash = polyglot_hash(pos);
        let first = self.find_first(hash)?;

        // Collect matching entries
        let mut moves = Vec::new();
        let mut i = first;
        while i < self.num_entries && self.entry_key(i) == hash {
            let w = self.entry_weight(i);
            if w > 0 {
                moves.push((self.entry_move(i), w));
            }
            i += 1;
        }

        if moves.is_empty() { return None; }

        let chosen_raw = if best_only {
            // Always pick the highest-weight move
            moves.iter().max_by_key(|&&(_, w)| w).unwrap().0
        } else {
            // Weighted random selection
            let total_weight: u64 = moves.iter().map(|&(_, w)| w as u64).sum();
            let mut pick = rng_val % total_weight;
            let mut chosen = moves[0].0;
            for &(raw, weight) in &moves {
                if pick < weight as u64 {
                    chosen = raw;
                    break;
                }
                pick -= weight as u64;
            }
            chosen
        };

        decode_poly_move(pos, chosen_raw)
    }
}

/// Decode a Polyglot raw_move (u16) into a clrsrc Move.
pub fn decode_poly_move(pos: &mut Position, raw: u16) -> Option<Move> {
    let to_file = (raw & 0x07) as u8;
    let to_row = ((raw >> 3) & 0x07) as u8;
    let from_file = ((raw >> 6) & 0x07) as u8;
    let from_row = ((raw >> 9) & 0x07) as u8;
    let promo_bits = ((raw >> 12) & 0x07) as u8;

    let from_sq = sq(from_file, from_row);
    let mut to_sq = sq(to_file, to_row);

    // Handle castling: Polyglot encodes king->rook, we need king->destination
    let piece = pos.mailbox[from_sq as usize];
    if piece != EMPTY && piece_type(piece) == KING {
        if from_file == 4 && to_file == 0 {
            to_sq = sq(2, from_row); // Queenside
        } else if from_file == 4 && to_file == 7 {
            to_sq = sq(6, from_row); // Kingside
        }
    }

    // Find matching legal move
    let mut list = MoveList::new();
    movegen::generate_all(pos, &mut list);

    for i in 0..list.len {
        let mv = list.moves[i];
        if mv.from() != from_sq || mv.to() != to_sq {
            continue;
        }

        if promo_bits > 0 {
            let flag = mv.flags();
            let promo_match = match promo_bits {
                1 => flag == Move::FLAG_KNIGHT_PROMO || flag == Move::FLAG_KNIGHT_PROMO_CAP,
                2 => flag == Move::FLAG_BISHOP_PROMO || flag == Move::FLAG_BISHOP_PROMO_CAP,
                3 => flag == Move::FLAG_ROOK_PROMO || flag == Move::FLAG_ROOK_PROMO_CAP,
                4 => flag == Move::FLAG_QUEEN_PROMO || flag == Move::FLAG_QUEEN_PROMO_CAP,
                _ => false,
            };
            if !promo_match {
                continue;
            }
        }

        // Verify legality
        let undo = pos.make_move(mv);
        let ksq = pos.king_sq(pos.side.flip());
        let legal = !attacks::is_attacked(&pos.pieces, &pos.colors, ksq, pos.side);
        pos.unmake_move(mv, undo);

        if legal {
            return Some(mv);
        }
    }

    None
}

/// Encode a clrsrc Move into a Polyglot raw_move (u16). Inverse of `decode_poly_move`.
/// Castling is stored king->rook (e1h1 / e1a1) per the Polyglot convention; promotions
/// use bits 12..15 (1=N, 2=B, 3=R, 4=Q). No position needed — the move's flags suffice.
pub fn encode_poly_move(mv: Move) -> u16 {
    let from = mv.from();
    let to = mv.to();
    let from_file = file_of(from) as u16;
    let from_rank = rank_of(from) as u16;
    let to_rank = rank_of(to) as u16;

    // Polyglot stores castling as king-captures-own-rook (king -> rook square), while clrsrc
    // encodes king -> destination. Remap the file to the rook's.
    let to_file = match mv.flags() {
        Move::FLAG_KING_CASTLE => 7,  // h-file rook
        Move::FLAG_QUEEN_CASTLE => 0, // a-file rook
        _ => file_of(to) as u16,
    };

    let promo: u16 = match mv.flags() {
        Move::FLAG_KNIGHT_PROMO | Move::FLAG_KNIGHT_PROMO_CAP => 1,
        Move::FLAG_BISHOP_PROMO | Move::FLAG_BISHOP_PROMO_CAP => 2,
        Move::FLAG_ROOK_PROMO | Move::FLAG_ROOK_PROMO_CAP => 3,
        Move::FLAG_QUEEN_PROMO | Move::FLAG_QUEEN_PROMO_CAP => 4,
        _ => 0,
    };

    to_file | (to_rank << 3) | (from_file << 6) | (from_rank << 9) | (promo << 12)
}

// ---- JBK2 Experience/Book reader (read-only consumer of jugernaut_v2.book) ----
// 32-byte little-endian header + 32-byte little-endian entries, sorted by (key, packed_move).
// Format is locked for version 2; a version mismatch is rejected rather than misread.

const JBK2_MAGIC: &[u8; 4] = b"JBK2";
const JBK2_HEADER_SIZE: usize = 32;
const JBK2_ENTRY_SIZE: usize = 32;
/// Sentinel for "score not set" (matches the writer's SCORE_NONE).
pub const EXP_SCORE_NONE: i16 = i16::MIN;

// `source`/`flags` bit values per the authoritative JBK2 spec (chess_engine/rust_engine/
// docs/exp_v2_format.md §7/§8). clrsrc never interprets these — they are carried through for
// the offline `expmerge`, so they MUST match the jugernaut definitions to merge correctly.
/// JBK2 §8 source bit5 = "clrsrc" (`SOURCE_CLRSRC`). Dedicated bit for this engine's live-learning
/// judgments, confirmed with the jugernaut instance 2026-05-27 (bit4=0x10 is already SOURCE_PGN —
/// reusing it would silently collide). When this bit is set, `clrsrc_score` (offset 28) is valid.
/// clrsrc never sets the Jugernaut bit (0x02) and never writes `jug_score`.
pub const EXP_SOURCE_ENGINE: u8 = 0b0010_0000;
/// JBK2 §8 source bit3 = SELFPLAY (entry came from an engine self-play game, not human/PGN data).
/// clrsrc-generated self-play book entries set `SELFPLAY | clrsrc` (0x28), matching the lichess-bot
/// harvest convention (coordinated with the bot instance 2026-05-27): the cp eval lives in
/// `clrsrc_score` so it survives `expmerge` (a bare 0x08 entry carries no per-source score).
#[allow(dead_code)] // consumed only by the working-tree self-play book tooling (not shipped)
pub const EXP_SOURCE_SELFPLAY: u8 = 0b0000_1000;
/// JBK2 §7 flags bit0 = VALIDATED (move verified legal / written by a completed search).
pub const EXP_FLAG_VALIDATED: u8 = 0x01;
/// JBK2 §7 flags bit4 = MATE_SCORE (the stored `score` is a mate score, not centipawns).
pub const EXP_FLAG_MATE: u8 = 0x10;
/// JBK2 §3 header flags bit1 = OVERLAY (this file is an append-only overlay, not the sorted main).
const JBK2_HEADER_FLAG_OVERLAY: u16 = 0x0002;
/// JBK2 §3 header flags bit0 = SORTED (the entries are sorted by (key, packed_move) — main file).
const JBK2_HEADER_FLAG_SORTED: u16 = 0x0001;
/// JBK2 §8 source bit1 = Jugernaut (analysis-engine judgment; owns `jug_score`).
const EXP_SOURCE_JUGERNAUT: u8 = 0x02;
/// JBK2 §8 source bit2 = Stockfish (oracle judgment; owns `sf_score`).
const EXP_SOURCE_STOCKFISH: u8 = 0x04;

#[derive(Clone, Copy, Debug)]
pub struct ExpEntry {
    pub key: u64,
    pub packed_move: u16,
    pub score: i16,
    pub depth: i16,
    pub count: u16,
    pub source: u8,
    pub flags: u8,
    pub wdl_w: u16,
    pub wdl_l: u16,
    pub nnue_eval: i16,
    pub jug_score: i16,
    pub sf_score: i16,
    /// Offset 28: foreign-engine (clrsrc) eval. Valid only when `source & EXP_SOURCE_ENGINE`.
    pub clrsrc_score: i16,
}

impl ExpEntry {
    fn from_bytes(b: &[u8]) -> ExpEntry {
        ExpEntry {
            key: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            packed_move: u16::from_le_bytes(b[8..10].try_into().unwrap()),
            score: i16::from_le_bytes(b[10..12].try_into().unwrap()),
            depth: i16::from_le_bytes(b[12..14].try_into().unwrap()),
            count: u16::from_le_bytes(b[14..16].try_into().unwrap()),
            source: b[16],
            flags: b[17],
            wdl_w: u16::from_le_bytes(b[18..20].try_into().unwrap()),
            wdl_l: u16::from_le_bytes(b[20..22].try_into().unwrap()),
            nnue_eval: i16::from_le_bytes(b[22..24].try_into().unwrap()),
            jug_score: i16::from_le_bytes(b[24..26].try_into().unwrap()),
            sf_score: i16::from_le_bytes(b[26..28].try_into().unwrap()),
            clrsrc_score: i16::from_le_bytes(b[28..30].try_into().unwrap()),
        }
    }

    fn to_bytes(&self) -> [u8; JBK2_ENTRY_SIZE] {
        let mut b = [0u8; JBK2_ENTRY_SIZE];
        b[0..8].copy_from_slice(&self.key.to_le_bytes());
        b[8..10].copy_from_slice(&self.packed_move.to_le_bytes());
        b[10..12].copy_from_slice(&self.score.to_le_bytes());
        b[12..14].copy_from_slice(&self.depth.to_le_bytes());
        b[14..16].copy_from_slice(&self.count.to_le_bytes());
        b[16] = self.source;
        b[17] = self.flags;
        b[18..20].copy_from_slice(&self.wdl_w.to_le_bytes());
        b[20..22].copy_from_slice(&self.wdl_l.to_le_bytes());
        b[22..24].copy_from_slice(&self.nnue_eval.to_le_bytes());
        b[24..26].copy_from_slice(&self.jug_score.to_le_bytes());
        b[26..28].copy_from_slice(&self.sf_score.to_le_bytes());
        b[28..30].copy_from_slice(&self.clrsrc_score.to_le_bytes());
        // bytes 30..32 = reserved u16 = 0
        b
    }
}

/// JBK2 book/experience file, read-only. Loaded fully into memory; binary-searched per probe.
pub struct ExpBook {
    data: Vec<u8>, // entries region only (header stripped)
    num_entries: usize,
    sorted: bool, // header SORTED flag — find_first/probe require this; false for overlays
}

impl ExpBook {
    /// Open a JBK2 file. Returns None on missing file, bad magic, or unsupported version.
    pub fn load(path: &str) -> Option<ExpBook> {
        let raw = std::fs::read(path).ok()?;
        if raw.len() < JBK2_HEADER_SIZE || &raw[0..4] != JBK2_MAGIC {
            eprintln!("info string exp: {} is not a JBK2 file", path);
            return None;
        }
        let version = u16::from_le_bytes(raw[4..6].try_into().unwrap());
        if version != 2 {
            eprintln!("info string exp: unsupported JBK2 version {} (expected 2)", version);
            return None;
        }
        let flags = u16::from_le_bytes(raw[6..8].try_into().unwrap());
        let sorted = flags & JBK2_HEADER_FLAG_SORTED != 0;
        let declared = u64::from_le_bytes(raw[8..16].try_into().unwrap()) as usize;
        let avail = (raw.len() - JBK2_HEADER_SIZE) / JBK2_ENTRY_SIZE;
        let num_entries = declared.min(avail);
        let data = raw[JBK2_HEADER_SIZE..JBK2_HEADER_SIZE + num_entries * JBK2_ENTRY_SIZE].to_vec();
        eprintln!(
            "info string exp: opened {} ({} entries, v{}, {})",
            path, num_entries, version,
            if sorted { "sorted" } else { "OVERLAY/unsorted" }
        );
        Some(ExpBook { data, num_entries, sorted })
    }

    pub fn entry_count(&self) -> usize {
        self.num_entries
    }

    /// All entries in file order (works for sorted main files and unsorted overlays alike).
    pub fn all_entries(&self) -> Vec<ExpEntry> {
        (0..self.num_entries).map(|i| self.entry(i)).collect()
    }

    #[inline]
    fn entry_key(&self, idx: usize) -> u64 {
        let off = idx * JBK2_ENTRY_SIZE;
        u64::from_le_bytes(self.data[off..off + 8].try_into().unwrap())
    }

    #[inline]
    fn entry(&self, idx: usize) -> ExpEntry {
        let off = idx * JBK2_ENTRY_SIZE;
        ExpEntry::from_bytes(&self.data[off..off + JBK2_ENTRY_SIZE])
    }

    /// First index whose key == `key`, or None.
    fn find_first(&self, key: u64) -> Option<usize> {
        // Binary search requires sorted entries. An overlay (OVERLAY flag, append-only,
        // unsorted) probed directly would yield silently wrong/missing hits. Refuse it and
        // warn once rather than return garbage — merge overlays into a sorted main
        // (write_sorted_main) before probing.
        if !self.sorted {
            use std::sync::atomic::{AtomicBool, Ordering};
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::Relaxed) {
                eprintln!("info string exp: WARNING probe on unsorted OVERLAY — no book moves used; merge into a sorted main first");
            }
            return None;
        }
        let mut lo = 0usize;
        let mut hi = self.num_entries;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.entry_key(mid) < key {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo < self.num_entries && self.entry_key(lo) == key {
            Some(lo)
        } else {
            None
        }
    }

    /// All entries for the position (same Polyglot key), in file order.
    pub fn probe(&self, pos: &Position) -> Vec<ExpEntry> {
        let key = polyglot_hash(pos);
        let mut out = Vec::new();
        if let Some(first) = self.find_first(key) {
            let mut i = first;
            while i < self.num_entries && self.entry_key(i) == key {
                out.push(self.entry(i));
                i += 1;
            }
        }
        out
    }

    /// Pick a move for the position. `variety`: 0=strict best, 1=±15cp, 2=±30cp tolerance.
    /// WDL filter drops statistically losing moves; a popularity (count) floor drops offbeat junk.
    /// variety 0: deterministic — highest depth, then count, then score.
    /// variety >0: uniform-random over the tolerance+WDL+popularity pool, seeded by `rng`
    /// (count-weighting is too peaked here — one mainline dominates — so uniform gives genuine
    /// opening spread, bounded in quality by the score cutoff, WDL filter, and count floor).
    pub fn probe_best(&self, pos: &mut Position, variety: u8, rng: u64) -> Option<Move> {
        let entries = self.probe(pos);
        if entries.is_empty() {
            return None;
        }
        let eff_variety = variety;
        let best_score = entries.iter().map(|e| e.score).max().unwrap();
        let cutoff = match eff_variety {
            1 => best_score.saturating_sub(15),
            2 => best_score.saturating_sub(30),
            _ => best_score,
        };
        let in_cutoff: Vec<&ExpEntry> = entries.iter().filter(|e| e.score >= cutoff).collect();
        // WDL filter: require loss-rate < 50% once we have a meaningful sample.
        let wdl_ok: Vec<&ExpEntry> = in_cutoff
            .iter()
            .copied()
            .filter(|e| {
                let total = e.wdl_w as i32 + e.wdl_l as i32;
                total < 6 || (e.wdl_l as i32) * 2 < total + 2
            })
            .collect();
        let pool = if wdl_ok.is_empty() { in_cutoff } else { wdl_ok };

        if eff_variety == 0 {
            // Deterministic best.
            let best = pool
                .iter()
                .max_by(|a, b| {
                    a.depth
                        .cmp(&b.depth)
                        .then(a.count.cmp(&b.count))
                        .then(a.score.cmp(&b.score))
                })
                .unwrap();
            return decode_poly_move(pos, best.packed_move);
        }

        // Popularity gate: polyglot-sourced entries carry score=0 (no real eval), so the score
        // cutoff alone lets offbeat junk (e.g. 1.g4) into the pool. Drop moves whose visit count
        // is below max_count/20 — keeps actually-played mainlines, removes sidelines.
        let max_count = pool.iter().map(|e| e.count).max().unwrap_or(0);
        let floor = max_count / 20;
        let quality: Vec<&ExpEntry> = pool.iter().copied().filter(|e| e.count >= floor).collect();
        let final_pool = if quality.is_empty() { pool } else { quality };

        // Mix the seed (splitmix64 finalizer): Windows clock granularity leaves the low bits of a
        // time-based seed near-constant, which would bias `rng % len` for small pools.
        let mut z = rng.wrapping_add(0x9E3779B97F4A7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;

        // Uniform-random over the qualifying pool for genuine opening variety.
        let idx = (z % final_pool.len() as u64) as usize;
        decode_poly_move(pos, final_pool[idx].packed_move)
    }
}

/// Merge a flat list of entries (main + overlay concatenated) per the JBK2 multi-source policy
/// (`exp_v2_format.md §11`). Groups by (key, packed_move); for each group:
/// - `source`/`flags` OR'd; `count`/`wdl_w`/`wdl_l` saturating-added.
/// - per-source scores (`jug_score` / `sf_score` / `clrsrc_score`) gated on the owning source bit,
///   higher per-source depth wins. Foreign engines never touch `jug_score`/`sf_score`.
/// - `depth` = max of the contributing per-source depths (0 if no score source).
/// - `nnue_eval` = first non-sentinel seen.
/// - `score` = canonical mirror. `clrsrc_first=false` → jug→sf→clrsrc→0 (jugernaut.book priority);
///   `clrsrc_first=true` → clrsrc→jug→sf→0 (clrsrc.exp home priority).
/// Returns entries sorted by (key, packed_move) ascending — ready for a sorted main file.
pub fn merge_entries(entries: &[ExpEntry], clrsrc_first: bool) -> Vec<ExpEntry> {
    use std::collections::BTreeMap;
    struct Acc {
        e: ExpEntry,
        jug_depth: i32,
        sf_depth: i32,
        clrsrc_depth: i32,
    }
    let mut map: BTreeMap<(u64, u16), Acc> = BTreeMap::new();
    for src in entries {
        let acc = map.entry((src.key, src.packed_move)).or_insert_with(|| Acc {
            e: ExpEntry {
                key: src.key,
                packed_move: src.packed_move,
                score: 0,
                depth: 0,
                count: 0,
                source: 0,
                flags: 0,
                wdl_w: 0,
                wdl_l: 0,
                nnue_eval: EXP_SCORE_NONE,
                jug_score: EXP_SCORE_NONE,
                sf_score: EXP_SCORE_NONE,
                clrsrc_score: EXP_SCORE_NONE,
            },
            jug_depth: -1,
            sf_depth: -1,
            clrsrc_depth: -1,
        });
        acc.e.source |= src.source;
        acc.e.flags |= src.flags;
        acc.e.count = acc.e.count.saturating_add(src.count);
        acc.e.wdl_w = acc.e.wdl_w.saturating_add(src.wdl_w);
        acc.e.wdl_l = acc.e.wdl_l.saturating_add(src.wdl_l);
        if acc.e.nnue_eval == EXP_SCORE_NONE && src.nnue_eval != EXP_SCORE_NONE {
            acc.e.nnue_eval = src.nnue_eval;
        }
        let d = src.depth as i32;
        if src.source & EXP_SOURCE_JUGERNAUT != 0 && src.jug_score != EXP_SCORE_NONE && d >= acc.jug_depth {
            acc.e.jug_score = src.jug_score;
            acc.jug_depth = d;
        }
        if src.source & EXP_SOURCE_STOCKFISH != 0 && src.sf_score != EXP_SCORE_NONE && d >= acc.sf_depth {
            acc.e.sf_score = src.sf_score;
            acc.sf_depth = d;
        }
        if src.source & EXP_SOURCE_ENGINE != 0 && src.clrsrc_score != EXP_SCORE_NONE && d >= acc.clrsrc_depth {
            acc.e.clrsrc_score = src.clrsrc_score;
            acc.clrsrc_depth = d;
        }
    }
    let mut out = Vec::with_capacity(map.len());
    for (_, acc) in map {
        let mut e = acc.e;
        e.depth = acc.jug_depth.max(acc.sf_depth).max(acc.clrsrc_depth).max(0) as i16;
        let mirror = if clrsrc_first {
            [e.clrsrc_score, e.jug_score, e.sf_score]
        } else {
            [e.jug_score, e.sf_score, e.clrsrc_score]
        };
        e.score = mirror.into_iter().find(|&s| s != EXP_SCORE_NONE).unwrap_or(0);
        out.push(e);
    }
    out
}

/// Write entries as a sorted JBK2 main file (header flag SORTED). Entries are sorted defensively
/// by (key, packed_move); `build_timestamp` is the current wall clock.
pub fn write_sorted_main(path: &str, entries: &[ExpEntry]) -> std::io::Result<()> {
    let mut sorted: Vec<&ExpEntry> = entries.iter().collect();
    sorted.sort_by(|a, b| a.key.cmp(&b.key).then(a.packed_move.cmp(&b.packed_move)));
    let mut buf = Vec::with_capacity(JBK2_HEADER_SIZE + sorted.len() * JBK2_ENTRY_SIZE);
    let mut hdr = [0u8; JBK2_HEADER_SIZE];
    hdr[0..4].copy_from_slice(JBK2_MAGIC);
    hdr[4..6].copy_from_slice(&2u16.to_le_bytes());
    hdr[6..8].copy_from_slice(&JBK2_HEADER_FLAG_SORTED.to_le_bytes());
    hdr[8..16].copy_from_slice(&(sorted.len() as u64).to_le_bytes());
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    hdr[16..24].copy_from_slice(&ts.to_le_bytes());
    buf.extend_from_slice(&hdr);
    for e in sorted {
        buf.extend_from_slice(&e.to_bytes());
    }
    // Atomic-ish: write to a temp file then rename over the target.
    let tmp = format!("{}.tmp", path);
    std::fs::write(&tmp, &buf)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Append-only writer for a JBK2 overlay file (`<exp>.overlay`). The engine pushes its deep
/// search judgments here; the overlay is merged into the main book offline by `expmerge`.
/// Kept fully separate from the read-only `ExpBook` so learning never mutates the shipped book.
/// The overlay is unsorted (header flags = 0); `expmerge` sorts on merge.
pub struct OverlayWriter {
    path: String,
    pending: Vec<ExpEntry>,
    on_disk: u64, // entries already persisted to the file (from its header)
}

impl OverlayWriter {
    /// Open (do not yet create) an overlay at `path`. Reads the existing entry count from the
    /// header if a valid overlay is already there, so subsequent flushes append rather than clobber.
    pub fn open(path: &str) -> OverlayWriter {
        let on_disk = match std::fs::read(path) {
            Ok(raw) if raw.len() >= JBK2_HEADER_SIZE && &raw[0..4] == JBK2_MAGIC => {
                u64::from_le_bytes(raw[8..16].try_into().unwrap())
            }
            _ => 0,
        };
        OverlayWriter { path: path.to_string(), pending: Vec::new(), on_disk }
    }

    pub fn push(&mut self, e: ExpEntry) {
        self.pending.push(e);
    }

    /// Append all pending entries to disk: create the file with a JBK2 header if missing,
    /// append the entries, then update the header's entry_count. Returns the number flushed.
    pub fn flush(&mut self) -> usize {
        if self.pending.is_empty() {
            return 0;
        }
        use std::io::{Seek, SeekFrom, Write};
        let mut f = match std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&self.path)
        {
            Ok(f) => f,
            Err(e) => {
                eprintln!("info string exp overlay open failed: {}", e);
                return 0;
            }
        };
        let len = f.metadata().map(|m| m.len()).unwrap_or(0);
        if len < JBK2_HEADER_SIZE as u64 {
            // Fresh file: write a 32-byte JBK2 header. flags = OVERLAY (unsorted, append-only).
            let mut hdr = [0u8; JBK2_HEADER_SIZE];
            hdr[0..4].copy_from_slice(JBK2_MAGIC);
            hdr[4..6].copy_from_slice(&2u16.to_le_bytes()); // version
            hdr[6..8].copy_from_slice(&JBK2_HEADER_FLAG_OVERLAY.to_le_bytes());
            // bytes 8..16 entry_count (set below), 16..24 build_timestamp, 24..32 reserved=0
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            hdr[16..24].copy_from_slice(&ts.to_le_bytes());
            if f.write_all(&hdr).is_err() {
                eprintln!("info string exp overlay header write failed");
                return 0;
            }
            self.on_disk = 0;
        }
        if f.seek(SeekFrom::End(0)).is_err() {
            return 0;
        }
        let n = self.pending.len();
        for e in self.pending.drain(..) {
            if f.write_all(&e.to_bytes()).is_err() {
                eprintln!("info string exp overlay entry write failed");
                break;
            }
        }
        self.on_disk += n as u64;
        if f.seek(SeekFrom::Start(8)).is_ok() {
            let _ = f.write_all(&self.on_disk.to_le_bytes());
        }
        let _ = f.flush();
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polyglot_hash_startpos_vector() {
        let pos = Position::startpos();
        assert_eq!(polyglot_hash(&pos), 0x463B96181691FC9C);
    }

    #[test]
    fn encode_kingside_castle_vector() {
        // e1->h1 (king captures own rook) = to h1 (file 7, rank 0), from e1 (file 4, rank 0).
        let mv = Move::new(sq(4, 0), sq(6, 0), Move::FLAG_KING_CASTLE);
        assert_eq!(encode_poly_move(mv), 0x0107);
    }

    #[test]
    fn encode_decode_roundtrip() {
        // Lookup tables are normally initialized in main(); do it here for the unit test.
        crate::zobrist::init();
        crate::magic::init();
        crate::attacks::init();
        // startpos, both-side castling, and positions with promotions available.
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
            "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1", // pawns on 2nd/7th: quiet + capture promos
        ];
        for fen in fens {
            let mut pos = Position::from_fen(fen).unwrap();
            let mut list = MoveList::new();
            movegen::generate_legal(&mut pos, &mut list);
            assert!(list.len > 0, "no legal moves in {}", fen);
            for i in 0..list.len {
                let mv = list.moves[i];
                let raw = encode_poly_move(mv);
                let decoded = decode_poly_move(&mut pos, raw);
                assert!(
                    decoded == Some(mv),
                    "roundtrip mismatch for {} (raw {:#06x}) in {}: got {:?}",
                    mv.to_uci(),
                    raw,
                    fen,
                    decoded.map(|m| m.to_uci())
                );
            }
        }
    }

    #[test]
    fn expmerge_matches_golden_fixture() {
        // Byte-conformance against the jugernaut reference merge (exp_v2::merge_with).
        // Fixture lives in the chess_engine tree; skip cleanly if not present here.
        let dir = r"P:\Projekte\chess_engine\postfach\fixtures";
        let book_p = format!(r"{}\merge_golden_book.bin", dir);
        let overlay_p = format!(r"{}\merge_golden_overlay.bin", dir);
        let expected_p = format!(r"{}\merge_golden_expected.bin", dir);
        let (book, overlay, expected) = match (
            ExpBook::load(&book_p),
            ExpBook::load(&overlay_p),
            std::fs::read(&expected_p),
        ) {
            (Some(b), Some(o), Ok(e)) => (b, o, e),
            _ => {
                eprintln!("golden fixture not present, skipping conformance test");
                return;
            }
        };
        let mut combined = book.all_entries();
        combined.extend(overlay.all_entries());
        // Fixture uses jugernaut.book priority (jug-first mirror).
        let merged = merge_entries(&combined, false);

        let out_p = std::env::temp_dir().join("clrsrc_expmerge_golden.bin");
        let out_s = out_p.to_str().unwrap();
        write_sorted_main(out_s, &merged).unwrap();
        let mut got = std::fs::read(out_s).unwrap();
        let _ = std::fs::remove_file(&out_p);

        let mut want = expected;
        // Zero the non-deterministic build_timestamp (header offset 16..24) on both sides.
        for buf in [&mut got, &mut want] {
            if buf.len() >= 24 {
                for b in &mut buf[16..24] {
                    *b = 0;
                }
            }
        }
        assert_eq!(got.len(), want.len(), "merged size differs from golden");
        assert_eq!(got, want, "merged bytes differ from golden fixture");
    }

    #[test]
    fn overlay_writer_append_and_count() {
        let dir = std::env::temp_dir();
        let path = dir.join("clrsrc_test_overlay.bin");
        let _ = std::fs::remove_file(&path);
        let p = path.to_str().unwrap();

        let mk = |key: u64| ExpEntry {
            key,
            packed_move: 0x0107,
            score: 25,
            depth: 20,
            count: 1,
            source: EXP_SOURCE_ENGINE,
            flags: EXP_FLAG_VALIDATED,
            wdl_w: 0,
            wdl_l: 0,
            nnue_eval: EXP_SCORE_NONE,
            jug_score: EXP_SCORE_NONE,
            sf_score: EXP_SCORE_NONE,
            clrsrc_score: 25,
        };

        // First flush creates the file with header + 2 entries.
        let mut w = OverlayWriter::open(p);
        w.push(mk(10));
        w.push(mk(20));
        assert_eq!(w.flush(), 2);
        assert_eq!(w.flush(), 0); // nothing pending

        // Re-open appends (does not clobber): 2 existing + 1 new = 3.
        let mut w2 = OverlayWriter::open(p);
        w2.push(mk(30));
        assert_eq!(w2.flush(), 1);

        // Read back via the standard reader: header count must be 3, keys present.
        let book = ExpBook::load(p).expect("overlay loads as JBK2");
        assert_eq!(book.entry_count(), 3);
        let mut pos = Position::startpos();
        // entries are unsorted in an overlay, but ExpBook::load does not re-sort; find_first
        // relies on sorted order, so just assert the raw count + first entry here.
        let _ = &mut pos;
        assert_eq!(book.entry(0).key, 10);
        assert_eq!(book.entry(2).key, 30);
        assert_eq!(book.entry(0).source, EXP_SOURCE_ENGINE);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn probe_on_unsorted_overlay_bails() {
        // Regression guard: an overlay (OVERLAY flag, unsorted) must NOT be probed via
        // binary search. Even with an entry keyed to the probed position, probe() must
        // return nothing (find_first bails on !sorted) rather than yield a wrong/missing
        // hit. Overlays are meant to be merged into a sorted main before probing.
        let dir = std::env::temp_dir();
        let path = dir.join("clrsrc_test_overlay_probe.bin");
        let _ = std::fs::remove_file(&path);
        let p = path.to_str().unwrap();

        let pos = Position::startpos();
        let mut w = OverlayWriter::open(p);
        w.push(ExpEntry {
            key: polyglot_hash(&pos),
            packed_move: 0x0107,
            score: 25,
            depth: 20,
            count: 1,
            source: EXP_SOURCE_ENGINE,
            flags: EXP_FLAG_VALIDATED,
            wdl_w: 9,
            wdl_l: 0,
            nnue_eval: EXP_SCORE_NONE,
            jug_score: EXP_SCORE_NONE,
            sf_score: EXP_SCORE_NONE,
            clrsrc_score: 25,
        });
        w.flush();

        let book = ExpBook::load(p).expect("overlay loads as JBK2");
        assert!(!book.sorted, "overlay must be flagged unsorted");
        assert_eq!(book.entry_count(), 1);
        // The key matches, but probe must bail because the book is unsorted.
        assert!(
            book.probe(&pos).is_empty(),
            "probe on an unsorted overlay must return nothing, not a wrong hit"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn exp_entry_roundtrip() {
        let e = ExpEntry {
            key: 0x463B96181691FC9C,
            packed_move: 0x0795,
            score: -123,
            depth: 27,
            count: 65535,
            source: 0b0000_1011,
            flags: 0x01,
            wdl_w: 12,
            wdl_l: 7,
            nnue_eval: i16::MIN,
            jug_score: -123,
            sf_score: 456,
            clrsrc_score: 88,
        };
        let bytes = e.to_bytes();
        let d = ExpEntry::from_bytes(&bytes);
        assert_eq!(d.key, e.key);
        assert_eq!(d.packed_move, e.packed_move);
        assert_eq!(d.score, e.score);
        assert_eq!(d.depth, e.depth);
        assert_eq!(d.count, e.count);
        assert_eq!(d.source, e.source);
        assert_eq!(d.flags, e.flags);
        assert_eq!(d.wdl_w, e.wdl_w);
        assert_eq!(d.wdl_l, e.wdl_l);
        assert_eq!(d.nnue_eval, e.nnue_eval);
        assert_eq!(d.jug_score, e.jug_score);
        assert_eq!(d.sf_score, e.sf_score);
        assert_eq!(d.clrsrc_score, e.clrsrc_score);
    }

    #[test]
    fn exp_book_load_search_and_version_reject() {
        let dir = std::env::temp_dir();
        let good = dir.join("clrsrc_test_good.book");
        let bad = dir.join("clrsrc_test_bad.book");

        let mut buf = Vec::new();
        buf.extend_from_slice(JBK2_MAGIC);
        buf.extend_from_slice(&2u16.to_le_bytes()); // version
        buf.extend_from_slice(&1u16.to_le_bytes()); // flags SORTED
        buf.extend_from_slice(&1u64.to_le_bytes()); // entry_count
        buf.extend_from_slice(&0u64.to_le_bytes()); // timestamp
        buf.extend_from_slice(&0u64.to_le_bytes()); // reserved
        let e = ExpEntry {
            key: 7, packed_move: 0x0795, score: 10, depth: 5, count: 3,
            source: 1, flags: 0, wdl_w: 0, wdl_l: 0,
            nnue_eval: EXP_SCORE_NONE, jug_score: EXP_SCORE_NONE, sf_score: EXP_SCORE_NONE,
            clrsrc_score: EXP_SCORE_NONE,
        };
        buf.extend_from_slice(&e.to_bytes());
        std::fs::write(&good, &buf).unwrap();

        let mut bad_buf = buf.clone();
        bad_buf[4] = 9; // version 9 -> must be rejected
        std::fs::write(&bad, &bad_buf).unwrap();

        let gb = ExpBook::load(good.to_str().unwrap()).expect("good book loads");
        assert_eq!(gb.entry_count(), 1);
        assert_eq!(gb.find_first(7), Some(0));
        assert_eq!(gb.find_first(8), None);
        assert_eq!(gb.entry(0).count, 3);

        assert!(ExpBook::load(bad.to_str().unwrap()).is_none());

        let _ = std::fs::remove_file(&good);
        let _ = std::fs::remove_file(&bad);
    }
}
